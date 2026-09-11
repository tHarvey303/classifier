import ast
import io
import json
import math
import operator
import os
import sys
import logging
import sqlite3
import tempfile
from datetime import datetime as dt
import numpy as np
import yaml
import boto3
from sqlalchemy import inspect
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import List, Dict, Optional, Tuple
from functools import wraps

# --- Load Environment Variables from .env ---
load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-key-please-change')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///local.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

REGISTRATION_PASSPHRASE = os.environ.get('REGISTRATION_PASSPHRASE')

# Comma-separated usernames that are always granted admin (e.g. ADMIN_USERNAMES=alice,bob)
ADMIN_USERNAMES = {
    name.strip() for name in os.environ.get('ADMIN_USERNAMES', '').split(',') if name.strip()
}

# R2 / S3 Config
R2_ENDPOINT = os.environ.get('R2_ENDPOINT_URL', '')
R2_KEY = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET = os.environ.get('R2_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')
R2_PUBLIC_DOMAIN = os.environ.get('R2_PUBLIC_DOMAIN', 'cdn.tharvey.space')

if R2_ENDPOINT and not R2_ENDPOINT.startswith("http"):
    R2_ENDPOINT = f"https://{R2_ENDPOINT}"

# --- Extensions ---
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# --- S3 Client ---
s3_client = boto3.client(
    's3',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_KEY,
    aws_secret_access_key=R2_SECRET,
    config=Config(signature_version='s3v4')
)

# --- Catalog Cache ---
# Keyed by folder name; value is either None (no catalog found) or a dict
# { 'columns': [...], 'rows': { 'ID_str': { col: val, ... }, ... } }
_catalog_cache: Dict[str, Optional[Dict]] = {}
# Raw astropy Table cache (needed for mask expression evaluation)
_catalog_table_cache: Dict[str, Optional[object]] = {}
# (key, ETag) of the R2 object the above two caches were parsed from, or
# (None, None) if no catalog.fits was found for that folder. Checked via a
# cheap head_object on every call so a catalog uploaded (or fixed) after the
# folder was first requested is picked up without needing a server restart.
_catalog_meta_cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
_CATALOG_META_UNSET = object()


class TableQueryParser:
    """Parse a filter expression string and apply it to an astropy Table as a boolean mask.

    Supports: comparisons (<, <=, >, >=, ==, !=), logic (&, |, ^), arithmetic (+, -,
    *, /), unary (~, -, +), grouping (), functions (log10, log, sqrt, abs, exp), and
    column indexing (col[i] → col[:, i]).
    """

    _OPERATORS = {
        ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge,
        ast.Eq: operator.eq, ast.NotEq: operator.ne,
        ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor,
        ast.Add: operator.add, ast.Sub: operator.sub,
        ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Invert: operator.invert, ast.USub: operator.neg, ast.UAdd: operator.pos,
    }
    _FUNCTIONS = {
        'log10': np.log10, 'log': np.log, 'sqrt': np.sqrt, 'abs': np.abs, 'exp': np.exp,
    }

    def __init__(self, table):
        self.table = table
        self._source = table

    def parse(self, expression: str) -> np.ndarray:
        """Return a boolean mask array of length len(table)."""
        if not expression or not expression.strip():
            return np.ones(len(self.table), dtype=bool)
        tree = ast.parse(expression, mode='eval')
        return np.asarray(self._eval(tree.body), dtype=bool)

    def _lookup_column(self, name: str) -> np.ndarray:
        src = self._source
        if name in src.colnames:
            return np.asarray(src[name])
        raise ValueError(f"Column '{name}' not found. Available: {list(src.colnames)}")

    def _eval_slice(self, node):
        return slice(
            self._eval(node.lower) if node.lower is not None else None,
            self._eval(node.upper) if node.upper is not None else None,
            self._eval(node.step) if node.step is not None else None,
        )

    def _eval_index(self, node):
        if hasattr(ast, 'Index') and isinstance(node, ast.Index):
            node = node.value
        if isinstance(node, ast.Slice):
            return self._eval_slice(node)
        if isinstance(node, ast.Tuple):
            return tuple(
                self._eval_slice(e) if isinstance(e, ast.Slice) else self._eval(e)
                for e in node.elts
            )
        return self._eval(node)

    def _eval(self, node):
        if isinstance(node, ast.BinOp):
            op = type(node.op)
            if op not in self._OPERATORS:
                raise ValueError(f"Unsupported operator: {op.__name__}")
            return self._OPERATORS[op](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.Compare):
            if len(node.ops) > 1:
                raise ValueError("Chained comparisons not supported; use (a < x) & (x < b).")
            op = type(node.ops[0])
            if op not in self._OPERATORS:
                raise ValueError(f"Unsupported comparison: {op.__name__}")
            return self._OPERATORS[op](self._eval(node.left), self._eval(node.comparators[0]))
        if isinstance(node, ast.UnaryOp):
            op = type(node.op)
            if op not in self._OPERATORS:
                raise ValueError(f"Unsupported unary op: {op.__name__}")
            return self._OPERATORS[op](self._eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are supported.")
            name = node.func.id
            if name not in self._FUNCTIONS:
                raise ValueError(f"Unknown function '{name}'. Available: {list(self._FUNCTIONS)}")
            return self._FUNCTIONS[name](*[self._eval(a) for a in node.args])
        if isinstance(node, ast.Subscript):
            base = np.asarray(self._eval(node.value))
            idx = self._eval_index(node.slice)
            if isinstance(node.value, ast.Name) and base.ndim >= 2:
                idx = (slice(None),) + (idx if isinstance(idx, tuple) else (idx,))
            return base[idx]
        if isinstance(node, ast.Name):
            return self._lookup_column(node.id)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Num):   # Python < 3.8
            return node.n
        if isinstance(node, ast.Str):   # Python < 3.8
            return node.s
        raise TypeError(f"Unsupported syntax node: {type(node).__name__}")


def _serialize_fits_value(val):
    """Convert a FITS/numpy scalar to a JSON-serialisable Python type."""
    if hasattr(val, 'item'):       # numpy scalar
        val = val.item()
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='replace').strip()
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(val, 6)
    return val


def load_catalog_for_folder(folder: str) -> Optional[Dict]:
    """
    Download catalog.fits from R2 for the given folder and parse it.
    The catalog must contain an 'ID' column whose values match image filenames
    (without extension).  Parsed results are cached in _catalog_cache, but the
    cache is validated on every call against a cheap head_object (key + ETag)
    so a catalog that is uploaded, replaced, or fixed after the folder was
    first requested is picked up on the next request instead of being stuck
    behind a stale in-memory result until the process restarts.
    """
    candidates = [
        f"classifier/{folder}/catalog.fits",
        f"{folder}/catalog.fits",
    ]

    found_key = None
    found_etag = None
    for key in candidates:
        try:
            head = s3_client.head_object(Bucket=BUCKET_NAME, Key=key)
            found_key = key
            found_etag = head.get('ETag')
            break
        except ClientError as e:
            code = e.response['Error']['Code']
            if code in ('NoSuchKey', '404'):
                continue
            logger.error(f"S3 error checking catalog {key}: {e}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error checking catalog {key}: {e}")
            continue

    current_meta = (found_key, found_etag)
    if _catalog_meta_cache.get(folder, _CATALOG_META_UNSET) == current_meta:
        return _catalog_cache.get(folder)

    if found_key is None:
        logger.info(f"No catalog found for folder '{folder}'")
        _catalog_cache[folder] = None
        _catalog_table_cache[folder] = None
        _catalog_meta_cache[folder] = current_meta
        return None

    try:
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=found_key)
        fits_bytes = obj['Body'].read()
        logger.info(f"Downloaded catalog from R2 key: {found_key} ({len(fits_bytes)} bytes)")
    except Exception as e:
        # Transient fetch error: don't cache anything so the next call retries.
        logger.error(f"Unexpected error fetching catalog {found_key}: {e}")
        return None

    try:
        from astropy.table import Table
        table = Table.read(io.BytesIO(fits_bytes))
    except Exception as e:
        logger.error(f"Failed to parse FITS catalog from {found_key}: {e}")
        _catalog_cache[folder] = None
        _catalog_table_cache[folder] = None
        _catalog_meta_cache[folder] = current_meta
        return None

    if 'ID' not in table.colnames:
        logger.warning(f"Catalog at {found_key} has no 'ID' column. Columns: {table.colnames}")
        _catalog_cache[folder] = None
        _catalog_table_cache[folder] = None
        _catalog_meta_cache[folder] = current_meta
        return None

    columns = list(table.colnames)
    rows: Dict[str, Dict] = {}
    for row in table:
        row_id = str(_serialize_fits_value(row['ID']))
        rows[row_id] = {col: _serialize_fits_value(row[col]) for col in columns}

    result = {'columns': columns, 'rows': rows}
    _catalog_cache[folder] = result
    _catalog_table_cache[folder] = table
    _catalog_meta_cache[folder] = current_meta
    logger.info(f"Catalog loaded for folder '{folder}': {len(rows)} rows, columns={columns}")
    return result


# --- Database Models ---
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)

class Classification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_key = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100))
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    __table_args__ = (db.UniqueConstraint('user_id', 'image_key', name='_user_image_uc'),)

class Assignment(db.Model):
    __tablename__ = 'assignments'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    folder = db.Column(db.String(500), nullable=False)
    start_index = db.Column(db.Integer, nullable=False)
    end_index = db.Column(db.Integer, nullable=False)
    note = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    __table_args__ = (db.UniqueConstraint('user_id', 'folder', name='_user_folder_assignment_uc'),)

class HiddenFolder(db.Model):
    __tablename__ = 'hidden_folders'
    id = db.Column(db.Integer, primary_key=True)
    folder = db.Column(db.String(500), unique=True, nullable=False)
    hidden_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class Tag(db.Model):
    """A user-defined label, shared by all users, for ad hoc grouping of images."""
    __tablename__ = 'tags'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    color = db.Column(db.String(20), nullable=False, default='#6b7280')
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())

class ImageTag(db.Model):
    """Membership of an image in a tag: one row per (tag, image); user_id records who added it."""
    __tablename__ = 'image_tags'
    id = db.Column(db.Integer, primary_key=True)
    tag_id = db.Column(db.Integer, db.ForeignKey('tags.id', ondelete='CASCADE'), nullable=False)
    image_key = db.Column(db.String(500), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    __table_args__ = (db.UniqueConstraint('tag_id', 'image_key', name='_tag_image_uc'),)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- HTML Templates ---

LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Source Classifier - Login</title>
    <style>
        body { font-family: -apple-system, sans-serif; display: flex; justify-content: center; align-items: center; height: 100vh; background: #f4f4f9; margin: 0; }
        .box { background: white; padding: 2.5rem; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); width: 320px; }
        h2 { margin-top: 0; color: #1f2937; text-align: center; margin-bottom: 1.5rem; }
        input { width: 100%; padding: 12px; margin: 8px 0; border: 1px solid #d1d5db; border-radius: 6px; box-sizing: border-box; font-size: 16px; transition: border 0.2s; }
        input:focus { border-color: #3b82f6; outline: none; }
        button { width: 100%; padding: 12px; background: #3b82f6; color: white; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; font-weight: 600; margin-top: 1rem; transition: background 0.2s; }
        button:hover { background: #2563eb; }
        .alert { background: #fee2e2; color: #991b1b; padding: 10px; border-radius: 6px; font-size: 0.9em; margin-bottom: 15px; text-align: center; }
        .helper { font-size: 0.85em; color: #6b7280; margin-top: 20px; text-align: center; line-height: 1.4; }
        .divider { border-top: 1px solid #e5e7eb; margin: 20px 0 10px 0; }
    </style>
</head>
<body>
    <div class="box">
        <h2>Source Classifier</h2>
        {% with messages = get_flashed_messages() %}
            {% if messages %}<div class="alert">{{ messages[0] }}</div>{% endif %}
        {% endwith %}
        <form method="POST">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            
            {% if require_passphrase %}
            <div class="divider"></div>
            <input type="password" name="passphrase" placeholder="Registration Key (New Users Only)">
            {% endif %}
            
            <button type="submit">Sign In / Register</button>
        </form>
        <div class="helper">
            Existing users: enter your credentials to log in.<br>
            {% if require_passphrase %}
            New users: enter a new username, password, and the Registration Key provided by the administrator.
            {% else %}
            New users: enter a new username and password to create an account.
            {% endif %}
        </div>
    </div>
</body>
</html>
"""

ADMIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Admin Dashboard</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: #f4f4f9; margin: 0; color: #1f2937; }
        .hdr { padding: 0.6rem 1.5rem; background: white; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; }
        .hdr h1 { font-size: 1rem; margin: 0; }
        .hdr a { color: #3b82f6; text-decoration: none; font-size: 0.875rem; margin-left: 1rem; }
        .main { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
        .tabs { display: flex; gap: 4px; margin-bottom: 1.5rem; }
        .tab-btn { padding: 8px 20px; border: 1px solid #d1d5db; background: white; border-radius: 6px; cursor: pointer; font-size: 0.875rem; font-weight: 500; color: #374151; }
        .tab-btn.active { background: #1f2937; color: white; border-color: #1f2937; }
        .tab-panel { display: none; }
        .tab-panel.active { display: block; }
        .card { background: white; border-radius: 10px; padding: 1.25rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 1.25rem; }
        .card h2 { margin: 0 0 1rem; font-size: 0.95rem; color: #374151; font-weight: 700; }
        table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
        th { text-align: left; padding: 8px 12px; background: #f9fafb; border-bottom: 2px solid #e5e7eb; color: #374151; font-weight: 600; }
        td { padding: 8px 12px; border-bottom: 1px solid #f3f4f6; vertical-align: middle; }
        tr:last-child td { border-bottom: none; }
        .btn { padding: 5px 12px; border-radius: 6px; cursor: pointer; font-size: 0.82rem; font-weight: 600; border: 1px solid; }
        .btn-blue { background: #3b82f6; color: white; border-color: #2563eb; }
        .btn-blue:hover { background: #2563eb; }
        .btn-red { background: white; color: #dc2626; border-color: #fca5a5; }
        .btn-red:hover { background: #fee2e2; }
        .btn-grey { background: white; color: #374151; border-color: #d1d5db; }
        .btn-grey:hover { background: #f3f4f6; }
        .btn-green { background: #059669; color: white; border-color: #047857; }
        .btn-green:hover { background: #047857; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
        .badge-admin { background: #fef3c7; color: #92400e; }
        .badge-user { background: #f3f4f6; color: #6b7280; }
        .badge-visible { background: #d1fae5; color: #065f46; }
        .badge-hidden { background: #fee2e2; color: #991b1b; }
        .field-row { display: flex; gap: 0.75rem; align-items: flex-end; flex-wrap: wrap; margin-bottom: 0.75rem; }
        .field { display: flex; flex-direction: column; gap: 3px; }
        .field label { font-size: 0.72rem; font-weight: 700; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
        .field input, .field select { padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.875rem; }
        .range-track { display: inline-block; width: 160px; height: 8px; background: #e5e7eb; border-radius: 999px; position: relative; overflow: hidden; vertical-align: middle; margin: 0 6px; }
        .range-fill { height: 100%; background: #3b82f6; border-radius: 999px; position: absolute; }
        .info-chip { font-size: 0.78rem; color: #6b7280; background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 4px; padding: 2px 8px; }
        .split-grid { display: flex; gap: 10px; flex-wrap: wrap; margin: 0.5rem 0 0.75rem; }
        .split-grid label { display: flex; align-items: center; gap: 5px; font-size: 0.875rem; cursor: pointer; }
        .split-preview-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; font-size: 0.85rem; gap: 8px; }
        .feedback { font-size: 0.82rem; min-height: 1.2em; margin-top: 4px; }
    </style>
</head>
<body>
<div class="hdr">
    <h1>&#9881; Admin Dashboard</h1>
    <div>
        <a href="{{ url_for('dashboard') }}">Team Dashboard</a>
        <a href="{{ url_for('index') }}">&#8592; Classifier</a>
    </div>
</div>
<div class="main">
    <div class="tabs">
        <button class="tab-btn active" onclick="showTab('users',this)">Users</button>
        <button class="tab-btn" onclick="showTab('assignments',this)">Assignments</button>
        <button class="tab-btn" onclick="showTab('folders',this)">Folders</button>
    </div>

    <div id="tab-users" class="tab-panel active">
        <div class="card">
            <h2>Users</h2>
            <table>
                <thead><tr><th>Username</th><th>Classifications</th><th>Role</th><th></th></tr></thead>
                <tbody id="users-tbody"><tr><td colspan="4" style="text-align:center;color:#9ca3af;">Loading&#8230;</td></tr></tbody>
            </table>
        </div>
    </div>

    <div id="tab-folders" class="tab-panel">
        <div class="card">
            <h2>Folders</h2>
            <p style="font-size:0.875rem;color:#6b7280;margin:0 0 1rem;">Hidden folders do not appear in the classifier folder list for non-admin users. Admins always see all folders.</p>
            <table>
                <thead><tr><th>Folder</th><th>Status</th><th></th></tr></thead>
                <tbody id="folders-tbody"><tr><td colspan="3" style="text-align:center;color:#9ca3af;">Loading&#8230;</td></tr></tbody>
            </table>
        </div>
    </div>

    <div id="tab-assignments" class="tab-panel">
        <div class="card">
            <div class="field-row" style="align-items:center;">
                <div class="field">
                    <label>Folder</label>
                    <select id="admin-folder-select" onchange="loadAssignmentFolder()" style="min-width:200px;">
                        <option value="" disabled selected>Select a folder&#8230;</option>
                    </select>
                </div>
                <span id="folder-img-count" class="info-chip" style="display:none;"></span>
            </div>
        </div>

        <div id="assignment-content" style="display:none;">
            <div class="card">
                <h2>Current Assignments</h2>
                <table>
                    <thead><tr><th>User</th><th>Range</th><th>Coverage</th><th>Count</th><th></th></tr></thead>
                    <tbody id="assignments-tbody"></tbody>
                </table>
            </div>

            <div class="card">
                <h2>Add / Update Assignment</h2>
                <div class="field-row">
                    <div class="field"><label>User</label><select id="assign-user"></select></div>
                    <div class="field"><label>Start (0-based)</label><input type="number" id="assign-start" min="0" style="width:110px;"></div>
                    <div class="field"><label>End (inclusive)</label><input type="number" id="assign-end" min="0" style="width:110px;"></div>
                    <div class="field"><label>Note (optional)</label><input type="text" id="assign-note" style="width:180px;" placeholder="e.g. overlap region"></div>
                    <button class="btn btn-blue" onclick="saveAssignment()" style="align-self:flex-end;">Save</button>
                </div>
                <div id="assign-feedback" class="feedback"></div>
            </div>

            <div class="card">
                <h2>Auto-split</h2>
                <p style="font-size:0.875rem;color:#6b7280;margin:0 0 1rem;">Divide images evenly across selected users with overlap for calibration. Replaces existing assignments.</p>
                <div style="font-size:0.72rem;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px;">Users to include</div>
                <div class="split-grid" id="split-users-grid"></div>
                <div class="field-row" style="align-items:flex-end;">
                    <div class="field"><label>Overlap %</label><input type="number" id="split-overlap" value="10" min="0" max="50" style="width:70px;"></div>
                    <button class="btn btn-grey" onclick="previewSplit()">Preview</button>
                    <button class="btn btn-green" id="btn-apply-split" onclick="applySplit()" style="display:none;">Apply all</button>
                </div>
                <div id="split-preview" style="display:none; margin-top:0.75rem; background:#f9fafb; border:1px solid #e5e7eb; border-radius:6px; padding:10px 12px;"></div>
            </div>
        </div>
    </div>
</div>

<script>
let adminUsers = [], adminFolders = [], adminFolderImages = [], splitPreviewData = [];

function showTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
}

async function init() {
    const [usersRes, foldersRes] = await Promise.all([fetch('/api/admin/users'), fetch('/api/admin/folders')]);
    adminUsers = await usersRes.json();
    adminFolders = await foldersRes.json();

    const tbody = document.getElementById('users-tbody');
    tbody.innerHTML = '';
    adminUsers.forEach(u => {
        const tr = tbody.insertRow();
        tr.innerHTML = `<td><b>${u.username}</b></td><td>${u.classification_count.toLocaleString()}</td>
            <td><span class="badge ${u.is_admin ? 'badge-admin' : 'badge-user'}" id="rbadge-${u.id}">${u.is_admin ? 'Admin' : 'User'}</span></td>
            <td><button class="btn btn-grey" id="rtbtn-${u.id}" onclick="toggleAdmin(${u.id})">${u.is_admin ? 'Revoke admin' : 'Make admin'}</button>
                <button class="btn btn-red" onclick="deleteUser(${u.id}, '${u.username}', ${u.classification_count})">Delete</button></td>`;
    });

    renderFolders();

    const folderSel = document.getElementById('admin-folder-select');
    adminFolders.forEach(f => {
        const o = document.createElement('option'); o.value = f.name;
        o.textContent = f.hidden ? `${f.name} (hidden)` : f.name;
        folderSel.appendChild(o);
    });

    const assignSel = document.getElementById('assign-user');
    const splitGrid = document.getElementById('split-users-grid');
    adminUsers.forEach(u => {
        const o = document.createElement('option'); o.value = u.id; o.textContent = u.username;
        assignSel.appendChild(o);
        const lbl = document.createElement('label');
        lbl.className = 'split-grid label';
        lbl.innerHTML = `<input type="checkbox" value="${u.id}" checked> ${u.username}`;
        splitGrid.appendChild(lbl);
    });
}

async function toggleAdmin(uid) {
    const res = await fetch(`/api/admin/users/${uid}/toggle_admin`, {method:'POST'});
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    document.getElementById(`rbadge-${uid}`).className = `badge ${data.is_admin ? 'badge-admin' : 'badge-user'}`;
    document.getElementById(`rbadge-${uid}`).textContent = data.is_admin ? 'Admin' : 'User';
    document.getElementById(`rtbtn-${uid}`).textContent = data.is_admin ? 'Revoke admin' : 'Make admin';
}

async function deleteUser(uid, username, count) {
    const warn = count > 0 ? `\\n\\nWARNING: their ${count.toLocaleString()} classification(s) and all assignments will be permanently deleted.` : '';
    if (!confirm(`Delete account "${username}"? This cannot be undone.${warn}`)) return;
    const res = await fetch(`/api/admin/users/${uid}`, {method:'DELETE'});
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    location.reload();
}

function renderFolders() {
    const tbody = document.getElementById('folders-tbody');
    tbody.innerHTML = '';
    if (!adminFolders.length) {
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:#9ca3af;">No folders found</td></tr>';
        return;
    }
    adminFolders.forEach(f => {
        const tr = tbody.insertRow();
        tr.innerHTML = `<td><b>${f.name}</b></td>
            <td><span class="badge ${f.hidden ? 'badge-hidden' : 'badge-visible'}">${f.hidden ? 'Hidden' : 'Visible'}</span></td>
            <td><button class="btn ${f.hidden ? 'btn-green' : 'btn-grey'}" onclick="toggleFolderHidden('${f.name}')">${f.hidden ? 'Unhide' : 'Hide'}</button></td>`;
    });
}

async function toggleFolderHidden(folder) {
    const res = await fetch('/api/admin/folders/toggle_hidden', {method:'POST',
        headers:{'Content-Type':'application/json'}, body:JSON.stringify({folder})});
    const data = await res.json();
    if (data.error) { alert(data.error); return; }
    const f = adminFolders.find(x => x.name === data.folder);
    if (f) f.hidden = data.hidden;
    renderFolders();
    const opt = [...document.getElementById('admin-folder-select').options].find(o => o.value === folder);
    if (opt) opt.textContent = data.hidden ? `${folder} (hidden)` : folder;
}

async function loadAssignmentFolder() {
    const folder = document.getElementById('admin-folder-select').value;
    if (!folder) return;
    const chip = document.getElementById('folder-img-count');
    chip.textContent = 'Loading&#8230;'; chip.style.display = 'inline';
    const res = await fetch(`/api/images?folder=${encodeURIComponent(folder)}&filter=All`);
    adminFolderImages = await res.json();
    chip.textContent = `${adminFolderImages.length} images`;
    document.getElementById('assign-end').placeholder = `0–${adminFolderImages.length-1}`;
    document.getElementById('assign-start').max = adminFolderImages.length - 1;
    document.getElementById('assign-end').max = adminFolderImages.length - 1;
    document.getElementById('assignment-content').style.display = 'block';
    await refreshAssignments(folder);
}

async function refreshAssignments(folder) {
    const res = await fetch(`/api/admin/assignments?folder=${encodeURIComponent(folder)}`);
    const data = await res.json();
    const tbody = document.getElementById('assignments-tbody');
    tbody.innerHTML = '';
    const total = adminFolderImages.length;
    if (!data.length) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#9ca3af;">No assignments yet</td></tr>';
        return;
    }
    data.forEach(a => {
        const count = a.end_index - a.start_index + 1;
        const pct = total > 0 ? count/total*100 : 0;
        const pctStart = total > 0 ? a.start_index/total*100 : 0;
        const tr = tbody.insertRow();
        tr.innerHTML = `<td><b>${a.username}</b>${a.note ? `<br><span style="font-size:0.75rem;color:#9ca3af;">${a.note}</span>` : ''}</td>
            <td>${a.start_index}–${a.end_index}</td>
            <td><div class="range-track" title="${pctStart.toFixed(0)}%–${(pctStart+pct).toFixed(0)}%">
                <div class="range-fill" style="left:${pctStart}%;width:${Math.max(1,pct)}%;"></div></div></td>
            <td>${count}</td>
            <td><button class="btn btn-red" onclick="deleteAssignment(${a.id})">Remove</button></td>`;
    });
}

async function saveAssignment() {
    const folder = document.getElementById('admin-folder-select').value;
    const userId = parseInt(document.getElementById('assign-user').value);
    const start = parseInt(document.getElementById('assign-start').value);
    const end = parseInt(document.getElementById('assign-end').value);
    const note = document.getElementById('assign-note').value.trim();
    const fb = document.getElementById('assign-feedback');
    if (isNaN(start)||isNaN(end)||start<0||end<start||end>=adminFolderImages.length) {
        fb.style.color='#dc2626'; fb.textContent=`Invalid range. Must be 0–${adminFolderImages.length-1} with start ≤ end.`; return;
    }
    const res = await fetch('/api/admin/assignments/save', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({user_id:userId, folder, start_index:start, end_index:end, note})});
    const data = await res.json();
    if (data.error) { fb.style.color='#dc2626'; fb.textContent=data.error; return; }
    fb.style.color='#059669'; fb.textContent='Saved.';
    setTimeout(()=>fb.textContent='',2000);
    refreshAssignments(folder);
}

async function deleteAssignment(id) {
    if (!confirm('Remove this assignment?')) return;
    const folder = document.getElementById('admin-folder-select').value;
    await fetch(`/api/admin/assignments/${id}`, {method:'DELETE'});
    refreshAssignments(folder);
}

function previewSplit() {
    const total = adminFolderImages.length;
    const overlapPct = Math.max(0, parseInt(document.getElementById('split-overlap').value)||0);
    const selected = [...document.querySelectorAll('#split-users-grid input:checked')]
        .map(cb => ({id:parseInt(cb.value), username:adminUsers.find(u=>u.id===parseInt(cb.value))?.username}));
    const previewEl = document.getElementById('split-preview');
    previewEl.style.display = 'block';
    if (selected.length < 2) { previewEl.innerHTML='<span style="color:#dc2626;font-size:0.85rem;">Select at least 2 users.</span>'; document.getElementById('btn-apply-split').style.display='none'; return; }
    const n = selected.length, chunkSize = Math.floor(total/n), overlapN = Math.max(0, Math.round(chunkSize*overlapPct/100));
    splitPreviewData = selected.map((u,i) => ({
        user_id:u.id, username:u.username,
        start:Math.max(0, i*chunkSize-overlapN),
        end:Math.min(total-1, (i+1)*chunkSize+overlapN-1)
    }));
    previewEl.innerHTML = splitPreviewData.map(a => {
        const cnt = a.end-a.start+1, pct=total>0?cnt/total*100:0, ps=total>0?a.start/total*100:0;
        return `<div class="split-preview-row"><b>${a.username}</b><span>${a.start}–${a.end}</span>
            <div class="range-track"><div class="range-fill" style="left:${ps}%;width:${pct}%;"></div></div>
            <span>${cnt} (${pct.toFixed(0)}%)</span></div>`;
    }).join('') + (overlapN>0?`<div style="margin-top:6px;font-size:0.78rem;color:#6b7280;">Adjacent overlap: ~${overlapN*2} images</div>`:'');
    document.getElementById('btn-apply-split').style.display = 'inline-block';
}

async function applySplit() {
    const folder = document.getElementById('admin-folder-select').value;
    for (const a of splitPreviewData) {
        await fetch('/api/admin/assignments/save', {method:'POST', headers:{'Content-Type':'application/json'},
            body:JSON.stringify({user_id:a.user_id, folder, start_index:a.start, end_index:a.end, note:'auto-split'})});
    }
    document.getElementById('split-preview').style.display='none';
    document.getElementById('btn-apply-split').style.display='none';
    splitPreviewData=[];
    refreshAssignments(folder);
}

init();
</script>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Team Dashboard</title>
    <style>
        body { font-family: -apple-system, sans-serif; background: #f4f4f9; margin: 0; color: #1f2937; }
        .hdr { padding: 0.6rem 1.5rem; background: white; border-bottom: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; }
        .hdr h1 { font-size: 1rem; margin: 0; }
        .hdr a { color: #3b82f6; text-decoration: none; font-size: 0.875rem; margin-left: 1rem; }
        .main { max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem; }
        .top-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
        .top-row h2 { margin: 0; font-size: 1rem; }
        .btn-refresh { padding: 5px 14px; border: 1px solid #d1d5db; background: white; border-radius: 6px; cursor: pointer; font-size: 0.85rem; }
        .btn-refresh:hover { background: #f3f4f6; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
        .sum-card { background: white; border-radius: 10px; padding: 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }
        .sum-card .num { font-size: 2rem; font-weight: 700; color: #1f2937; line-height: 1; }
        .sum-card .lbl { font-size: 0.75rem; color: #6b7280; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
        .folder-card { background: white; border-radius: 10px; padding: 1.25rem 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); margin-bottom: 1rem; }
        .folder-card h3 { margin: 0 0 0.75rem; font-size: 0.95rem; display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }
        .agree-badge { display: inline-flex; align-items: center; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
        .agree-hi { background: #d1fae5; color: #065f46; }
        .agree-mid { background: #fef3c7; color: #92400e; }
        .agree-lo { background: #fee2e2; color: #991b1b; }
        .agree-none { background: #f3f4f6; color: #6b7280; }
        .folder-meta { font-size: 0.78rem; color: #9ca3af; margin-left: auto; }
        .bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; font-size: 0.875rem; }
        .bar-label { width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #374151; }
        .bar-track { flex: 1; height: 10px; background: #e5e7eb; border-radius: 999px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 999px; }
        .bar-count { width: 45px; text-align: right; color: #6b7280; font-size: 0.8rem; }
        .cat-pills { margin-top: 6px; }
        .cat-pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; background: #f3f4f6; color: #374151; margin: 2px 2px 0 0; }
        .no-data { text-align: center; padding: 3rem; color: #9ca3af; }
    </style>
</head>
<body>
<div class="hdr">
    <h1>&#128202; Team Dashboard</h1>
    <div>
        {% if is_admin %}<a href="{{ url_for('admin') }}" style="color:#f59e0b;">&#9881; Admin</a>{% endif %}
        <a href="{{ url_for('index') }}">&#8592; Classifier</a>
    </div>
</div>
<div class="main">
    <div class="top-row">
        <h2>Overview</h2>
        <button class="btn-refresh" onclick="load()">&#8635; Refresh</button>
    </div>
    <div class="summary-grid">
        <div class="sum-card"><div class="num" id="s-total">—</div><div class="lbl">Classifications</div></div>
        <div class="sum-card"><div class="num" id="s-users">—</div><div class="lbl">Active users</div></div>
        <div class="sum-card"><div class="num" id="s-folders">—</div><div class="lbl">Folders</div></div>
        <div class="sum-card"><div class="num" id="s-agree">—</div><div class="lbl">Avg agreement</div></div>
    </div>
    <div class="top-row"><h2>Per-folder progress</h2></div>
    <div id="folder-list"><div class="no-data">Loading&#8230;</div></div>
</div>
<script>
const PALETTE = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];

async function load() {
    document.getElementById('folder-list').innerHTML = '<div class="no-data">Loading&#8230;</div>';
    const data = await (await fetch('/api/dashboard/summary')).json();
    const userTotals = data.user_totals;
    const allUsers = Object.keys(userTotals);
    const userColor = {};
    allUsers.forEach((u,i) => userColor[u] = PALETTE[i % PALETTE.length]);

    const total = allUsers.reduce((s,u) => s + userTotals[u].total, 0);
    document.getElementById('s-total').textContent = total.toLocaleString();
    document.getElementById('s-users').textContent = allUsers.length;
    document.getElementById('s-folders').textContent = data.folders.length;

    const agreeFolders = data.folders.filter(f => f.agree_rate !== null);
    const avgAgree = agreeFolders.length
        ? agreeFolders.reduce((s,f)=>s+f.agree_rate,0)/agreeFolders.length : null;
    document.getElementById('s-agree').textContent = avgAgree !== null ? (avgAgree*100).toFixed(0)+'%' : '—';

    const listEl = document.getElementById('folder-list');
    if (!data.folders.length) { listEl.innerHTML='<div class="no-data">No classifications yet.</div>'; return; }
    listEl.innerHTML = '';

    data.folders.forEach(f => {
        const ar = f.agree_rate;
        const [agreeClass, agreeText] = ar===null ? ['agree-none','No multi-user data'] :
            ar>=0.8 ? ['agree-hi', (ar*100).toFixed(0)+'% agreement'] :
            ar>=0.5 ? ['agree-mid', (ar*100).toFixed(0)+'% agreement'] :
                       ['agree-lo', (ar*100).toFixed(0)+'% agreement'];
        const maxCount = Math.max(1, ...Object.values(f.users));
        const barsHtml = Object.entries(f.users).map(([uname, cnt]) =>
            `<div class="bar-row">
                <div class="bar-label" title="${uname}">${uname}</div>
                <div class="bar-track"><div class="bar-fill" style="width:${cnt/maxCount*100}%;background:${userColor[uname]||'#3b82f6'};"></div></div>
                <div class="bar-count">${cnt.toLocaleString()}</div>
            </div>`).join('');
        const catHtml = Object.entries(f.category_counts||{}).sort((a,b)=>b[1]-a[1])
            .map(([c,n]) => `<span class="cat-pill">${c}: ${n}</span>`).join('');

        const card = document.createElement('div');
        card.className = 'folder-card';
        card.innerHTML = `<h3>
                <span>&#128193; ${f.name}</span>
                <span class="agree-badge ${agreeClass}">${agreeText}</span>
                <span class="folder-meta">${f.total_classified_images} images classified</span>
            </h3>
            <div>${barsHtml}</div>
            ${catHtml ? `<div class="cat-pills">${catHtml}</div>` : ''}`;
        listEl.appendChild(card);
    });
}

load();
</script>
</body>
</html>
"""

APP_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Source Classifier</title>
    <script src="https://cdn.plot.ly/plotly-basic-2.35.2.min.js"></script>
    <style>
        :root { --bg-color: #f4f4f9; --panel-bg: #ffffff; --accent: #3b82f6; --text-main: #1f2937; --text-sub: #6b7280; }
        body { margin: 0; font-family: -apple-system, sans-serif; background: var(--bg-color); height: 100vh; display: flex; flex-direction: column; }
        
        .header { padding: 0.5rem 1rem; background: white; border-bottom: 1px solid #ddd; display: flex; justify-content: space-between; align-items: center; }
        .header h1 { font-size: 1rem; margin: 0; color: var(--text-main); }
        .user-info { font-size: 0.9rem; }
        .user-info a { color: var(--accent); text-decoration: none; margin-left: 10px; font-weight: 500; }

        .container { display: flex; flex: 1; overflow: hidden; }
        
        /* Image Area */
        .image-area { flex: 3; background: #e5e7eb; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; }
        .image-area img { max-width: 95%; max-height: 95%; object-fit: contain; transition: transform 0.1s; }
        
        /* Sidebar */
        .sidebar { flex: 1; min-width: 320px; max-width: 400px; background: var(--panel-bg); border-left: 1px solid #d1d5db; padding: 2rem; display: flex; flex-direction: column; overflow-y: auto; }
        
        h2 { margin: 0 0 0.5rem 0; color: var(--text-main); font-size: 1.1rem; word-break: break-all; }
        .meta { color: var(--text-sub); font-size: 0.875rem; margin-bottom: 1rem; }
        .section-label { font-weight: 600; color: var(--text-main); margin-bottom: 0.5rem; display: block; font-size: 0.9rem; }
        
        .dropdown { width: 100%; padding: 0.6rem; margin-bottom: 1.5rem; border: 1px solid #d1d5db; border-radius: 0.5rem; font-size: 0.95rem; background: white; }
        
        .search-box { width: 100%; padding: 0.6rem; margin-bottom: 1.5rem; border: 1px solid #d1d5db; border-radius: 0.5rem; font-size: 0.95rem; box-sizing: border-box; }
        
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.5rem; }
        .btn-cat { padding: 0.75rem; border: 1px solid #d1d5db; background: white; border-radius: 0.5rem; cursor: pointer; text-align: left; color: var(--text-main); transition: all 0.2s; }
        .btn-cat:hover { background: #f3f4f6; border-color: #9ca3af; }
        .btn-cat.active { background: var(--accent); color: white; border-color: var(--accent); }
        
        .btn-secondary { width: 100%; padding: 8px; background: #e5e7eb; color: #374151; border: 1px solid #d1d5db; border-radius: 6px; cursor: pointer; margin-bottom: 10px; font-weight: 600; font-size: 0.9rem; }
        .btn-secondary:hover { background: #d1d5db; }
        .btn-secondary.active { background: #fef3c7; border-color: #f59e0b; color: #92400e; }
        /* Tags */
        .tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 0.5rem; min-height: 28px; }
        .tag-chip { display: inline-flex; align-items: center; gap: 5px; padding: 4px 10px; border-radius: 999px; border: 1px solid; font-size: 0.8rem; cursor: pointer; background: white; user-select: none; line-height: 1.2; transition: background 0.15s, color 0.15s; }
        .tag-chip:hover { filter: brightness(0.93); }
        .tag-chip.active { color: white; }
        .tag-chip.disabled { cursor: default; opacity: 0.7; }
        .tag-chip.disabled:hover { filter: none; }
        .tag-chip .tag-count { opacity: 0.7; font-size: 0.72rem; }
        .tag-chip .tag-icon { margin-left: 2px; padding: 0 3px; border-radius: 3px; font-size: 0.85rem; opacity: 0.75; }
        .tag-chip .tag-icon:hover { opacity: 1; background: rgba(0,0,0,0.15); }
        .tag-empty { color: var(--text-sub); font-size: 0.82rem; font-style: italic; }
        .tag-new-row { display: flex; gap: 6px; margin-bottom: 1rem; }
        .tag-new-row input { flex: 1; min-width: 0; padding: 6px 10px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.85rem; }
        .tag-new-row button { padding: 6px 12px; background: #e5e7eb; color: #374151; border: 1px solid #d1d5db; border-radius: 6px; cursor: pointer; font-weight: 600; font-size: 0.85rem; white-space: nowrap; }
        .tag-new-row button:hover { background: #d1d5db; }
        .tag-manage-box { background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 8px 10px; margin-bottom: 1rem; font-size: 0.82rem; color: #92400e; display: none; }
        .tag-manage-box .row { display: flex; gap: 6px; align-items: center; margin-top: 6px; }
        .tag-manage-box select { flex: 1; min-width: 0; padding: 5px 8px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.82rem; background: white; }
        .tag-manage-box button { padding: 5px 10px; border: 1px solid #d1d5db; border-radius: 6px; background: white; cursor: pointer; font-size: 0.8rem; font-weight: 600; color: #374151; white-space: nowrap; }
        .tag-manage-box button:hover { background: #f3f4f6; }
        
        .stats-box { background: #f3f4f6; padding: 10px; border-radius: 6px; margin-bottom: 15px; font-size: 0.9em; display: none; border: 1px solid #e5e7eb; }
        .stat-row { display: flex; justify-content: space-between; margin-bottom: 4px; padding-bottom: 4px; border-bottom: 1px dashed #d1d5db; }
        .stat-row:last-child { border-bottom: none; }
        
        .notes-area { width: 100%; height: 100px; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 0.5rem; resize: vertical; margin-bottom: 1.5rem; box-sizing: border-box; font-family: inherit; }
        
        .nav-row { display: flex; gap: 1rem; margin-top: auto; }
        .btn-nav { flex: 1; padding: 0.75rem; background: #1f2937; color: white; border: none; border-radius: 0.5rem; cursor: pointer; font-weight: 600; }
        .btn-nav:hover { background: #374151; }

        #toast { position: fixed; top: 20px; right: 20px; background: #10b981; color: white; padding: 10px 20px; border-radius: 5px; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 99; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }

        /* Catalog section */
        .catalog-sort-row { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
        .catalog-sort-row select { margin-bottom: 0; }
        .catalog-sort-row select:first-child { flex: 1; }
        .catalog-sort-row select:last-child { width: auto; }
        .filter-range-row { display: flex; gap: 0.4rem; align-items: center; font-size: 0.85rem; color: var(--text-sub); margin-bottom: 1rem; }
        .filter-range-input { flex: 1; min-width: 0; padding: 5px 6px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.85rem; }
        .mask-expr-row { display: flex; gap: 0.4rem; align-items: center; margin-bottom: 0.4rem; }
        .mask-expr-input { display: block; width: 100%; box-sizing: border-box; padding: 5px 6px; border: 1px solid #d1d5db; border-radius: 4px; font-size: 0.82rem; font-family: monospace; margin-bottom: 0.4rem; }
        .mask-expr-input.error { border-color: #dc2626; }
        .mask-expr-status { font-size: 0.78rem; color: var(--text-sub); margin-bottom: 0.8rem; min-height: 1.1em; }
        .mask-expr-status.err { color: #dc2626; }
        .catalog-props-box { background: #f3f4f6; border: 1px solid #e5e7eb; border-radius: 6px; padding: 8px 10px; margin-bottom: 1rem; font-size: 0.8em; min-height: 120px; max-height: 350px; overflow-y: auto; display: none; }
        .catalog-props-box .props-title { font-weight: 700; color: #374151; margin-bottom: 4px; }
        .prop-row { display: flex; justify-content: space-between; padding: 2px 0; border-bottom: 1px dashed #e5e7eb; }
        .prop-row:last-child { border-bottom: none; }
        .prop-key { color: var(--text-sub); }
        .prop-val { font-weight: 600; text-align: right; margin-left: 8px; }
        .catalog-loading { color: var(--text-sub); font-size: 0.85em; margin-bottom: 1rem; font-style: italic; }

        /* Scatter plot modal */
        #scatter-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:200; align-items:center; justify-content:center; }
        .scatter-panel { background:white; border-radius:12px; padding:1.25rem 1.5rem; width:88vw; height:88vh; max-width:1200px; display:flex; flex-direction:column; gap:0.75rem; box-shadow:0 25px 50px rgba(0,0,0,0.4); }
        .scatter-header { display:flex; justify-content:space-between; align-items:center; }
        .scatter-header h3 { margin:0; font-size:1.05rem; color:var(--text-main); }
        .scatter-point-count { font-size:0.8rem; color:var(--text-sub); background:#f3f4f6; border:1px solid #e5e7eb; border-radius:999px; padding:2px 10px; margin-left:0.75rem; }
        .scatter-controls { display:flex; gap:0.75rem; flex-wrap:wrap; align-items:flex-end; background:#f9fafb; padding:0.6rem 0.8rem; border-radius:8px; border:1px solid #e5e7eb; }
        .scatter-ctrl { display:flex; flex-direction:column; gap:3px; }
        .scatter-ctrl label { font-size:0.75rem; font-weight:600; color:var(--text-sub); text-transform:uppercase; letter-spacing:0.04em; }
        .scatter-ctrl select { padding:5px 8px; border:1px solid #d1d5db; border-radius:6px; font-size:0.9rem; background:white; }
        .scatter-ctrl-check { display:flex; align-items:center; gap:4px; font-size:0.85rem; color:var(--text-main); padding-bottom:2px; }
        #scatter-plot { flex:1; min-height:0; }
        .btn-close { background:none; border:1px solid #d1d5db; border-radius:6px; padding:4px 10px; cursor:pointer; font-size:1.1rem; color:#6b7280; }
        .btn-close:hover { background:#f3f4f6; }
        .btn-scatter { width:100%; padding:8px; background:#1f2937; color:white; border:none; border-radius:6px; cursor:pointer; margin-bottom:10px; font-weight:600; font-size:0.9rem; }
        .btn-scatter:hover { background:#374151; }
        .btn-sky { width:100%; padding:8px; background:#0f766e; color:white; border:none; border-radius:6px; cursor:pointer; margin-bottom:10px; font-weight:600; font-size:0.9rem; }
        .btn-sky:hover { background:#115e59; }
        .btn-sky.active { background:#134e4a; }
        .btn-lf { width:100%; padding:8px; background:#9333ea; color:white; border:none; border-radius:6px; cursor:pointer; margin-bottom:10px; font-weight:600; font-size:0.9rem; }
        .btn-lf:hover { background:#7e22ce; }

        /* Luminosity-function modal */
        #lf-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:200; align-items:center; justify-content:center; }
        .lf-panel { background:white; border-radius:12px; padding:1.25rem 1.5rem; width:90vw; height:90vh; max-width:1300px; display:flex; flex-direction:column; gap:0.75rem; box-shadow:0 25px 50px rgba(0,0,0,0.4); }
        .lf-body { flex:1; min-height:0; display:flex; gap:0.9rem; }
        #lf-plot { flex:1; min-height:0; }
        .lf-side { width:260px; flex-shrink:0; border-left:1px solid #e5e7eb; padding-left:0.9rem; display:flex; flex-direction:column; min-height:0; }
        .lf-side h4 { margin:0 0 0.5rem; font-size:0.9rem; color:var(--text-main); }
        .lf-gal-list { flex:1; overflow-y:auto; font-size:0.82rem; }
        .lf-gal-row { display:flex; justify-content:space-between; gap:6px; padding:4px 6px; border-radius:4px; cursor:pointer; }
        .lf-gal-row:hover { background:#f3e8ff; }
        .lf-gal-row .gid { font-family:monospace; color:#6b21a8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .lf-gal-row .gval { color:var(--text-sub); white-space:nowrap; }
        .lf-controls { display:flex; gap:0.7rem; flex-wrap:wrap; align-items:flex-end; background:#f9fafb; padding:0.6rem 0.8rem; border-radius:8px; border:1px solid #e5e7eb; }
        .lf-ctrl { display:flex; flex-direction:column; gap:3px; }
        .lf-ctrl label { font-size:0.72rem; font-weight:600; color:var(--text-sub); text-transform:uppercase; letter-spacing:0.04em; }
        .lf-ctrl select, .lf-ctrl input { padding:5px 8px; border:1px solid #d1d5db; border-radius:6px; font-size:0.88rem; background:white; }
        .lf-ctrl input.num { width:80px; }
        .lf-ctrl input.expr { width:200px; font-family:monospace; font-size:0.8rem; }
        .lf-ctrl-check { display:flex; align-items:center; gap:4px; font-size:0.85rem; color:var(--text-main); padding-bottom:4px; }
        .lf-status { font-size:0.8rem; color:var(--text-sub); min-height:1.1em; }
        .lf-status.err { color:#dc2626; }
        .lf-meta { font-size:0.78rem; color:var(--text-sub); }

        /* Sky position overlay */
        #sky-overlay { position:absolute; inset:0; background:#f9fafb; z-index:10; display:flex; flex-direction:column; padding:0.75rem 1rem; transform:translateX(-100%); transition:transform 0.25s ease; }
        #sky-overlay.open { transform:translateX(0); }
        #sky-overlay-header { display:flex; align-items:center; margin-bottom:0.5rem; }
        #sky-overlay-title { font-weight:700; font-size:0.95rem; color:#1f2937; }

        /* Export section */
        #export-section { margin-bottom:1.5rem; display:none; }
        .export-row { display:flex; gap:0.5rem; align-items:center; flex-wrap:wrap; margin-top:0.4rem; }
        .btn-export { padding:6px 14px; border:1px solid #d1d5db; background:white; border-radius:6px; cursor:pointer; font-size:0.85rem; font-weight:600; color:var(--text-main); }
        .btn-export:hover { background:#f3f4f6; border-color:#9ca3af; }
        .export-check { font-size:0.82rem; color:var(--text-sub); display:flex; align-items:center; gap:4px; margin-left:auto; }

        /* Compare modal */
        #compare-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:200; align-items:center; justify-content:center; }
        .compare-panel { background:white; border-radius:12px; padding:1.25rem 1.5rem; width:90vw; height:88vh; max-width:1400px; display:flex; flex-direction:column; gap:0.75rem; box-shadow:0 25px 50px rgba(0,0,0,0.4); }
        .compare-header { display:flex; justify-content:space-between; align-items:center; }
        .compare-header h3 { margin:0; font-size:1.05rem; color:var(--text-main); }
        .compare-controls { display:flex; gap:0.75rem; flex-wrap:wrap; align-items:flex-end; background:#f9fafb; padding:0.6rem 0.8rem; border-radius:8px; border:1px solid #e5e7eb; }
        #compare-table-wrapper { flex:1; min-height:0; overflow:auto; border:1px solid #e5e7eb; border-radius:6px; }
        #compare-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
        #compare-table th { position:sticky; top:0; background:#f9fafb; border-bottom:2px solid #e5e7eb; padding:8px 10px; text-align:left; font-weight:600; color:#374151; white-space:nowrap; z-index:1; }
        #compare-table td { padding:6px 10px; border-bottom:1px solid #f3f4f6; }
        #compare-table tbody tr { cursor:pointer; }
        #compare-table tbody tr:hover td { background:#f0f9ff; }
        .compare-cat-cell { border-radius:4px; padding:2px 8px; font-size:0.8rem; font-weight:600; white-space:nowrap; }
        .compare-agree { color:#059669; font-weight:700; }
        .compare-disagree { color:#dc2626; font-weight:700; }
        .btn-compare { width:100%; padding:8px; background:#4f46e5; color:white; border:none; border-radius:6px; cursor:pointer; margin-bottom:10px; font-weight:600; font-size:0.9rem; }
        .btn-compare:hover { background:#4338ca; }

        /* Import modal */
        #import-modal { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.75); z-index:200; align-items:center; justify-content:center; }
        .import-panel { background:white; border-radius:12px; padding:1.5rem; width:90vw; max-width:600px; max-height:85vh; display:flex; flex-direction:column; gap:1rem; box-shadow:0 25px 50px rgba(0,0,0,0.4); overflow-y:auto; }
        .import-header { display:flex; justify-content:space-between; align-items:center; flex-shrink:0; }
        .import-header h3 { margin:0; font-size:1.05rem; color:var(--text-main); }
        #import-dropzone { border:2px dashed #d1d5db; border-radius:8px; padding:2.5rem 1rem; text-align:center; cursor:pointer; color:#6b7280; transition:border-color 0.2s, background 0.2s; }
        #import-dropzone:hover { border-color:#3b82f6; background:#eff6ff; color:#2563eb; }
        .import-stat { display:flex; justify-content:space-between; align-items:center; padding:6px 10px; background:#f9fafb; border-radius:6px; font-size:0.9rem; margin-bottom:4px; }
        .import-stat b { color:var(--text-main); }
        .import-conflict-row { display:flex; gap:8px; padding:4px 0; font-size:0.8rem; border-bottom:1px dashed #f3f4f6; align-items:center; }
        #import-preview-rows { max-height:200px; overflow-y:auto; background:#f9fafb; border-radius:6px; padding:8px 10px; margin-top:8px; }
        .import-conflict-opts label { display:flex; align-items:center; gap:8px; margin-bottom:8px; cursor:pointer; font-size:0.9rem; }
        .btn-import-link { background:none; border:none; color:#059669; font-size:0.82rem; font-weight:600; cursor:pointer; padding:0; text-decoration:underline; }
        .btn-import-link:hover { color:#047857; }
    </style>
</head>
<body>

<div class="header">
    <h1>Source Classifier</h1>
    <div class="user-info">
        <b>{{ current_user.username }}</b>
        <a href="{{ url_for('dashboard') }}">Dashboard</a>
        {% if current_user.is_admin %}<a href="{{ url_for('admin') }}" style="color:#f59e0b;">Admin</a>{% endif %}
        <a href="/api/db_backup">Backup (JSON)</a>
        {% if is_sqlite %}<a href="/api/db_backup?format=sqlite">Backup (.db)</a>{% endif %}
        <a href="{{ url_for('logout') }}">Logout</a>
    </div>
</div>

<div class="container">
    <div class="image-area">
        <img id="main-image" src="" alt="Select a folder to begin">
        <div id="sky-overlay">
            <div id="sky-overlay-header">
                <span id="sky-overlay-title">Sky Position</span>
                <button class="btn-close" onclick="closeSkyPlot()" style="margin-left:auto;">&#10005;</button>
            </div>
            <div id="sky-plot-div" style="flex:1;min-height:0;"></div>
        </div>
    </div>
    
    <div class="sidebar">
        <h2 id="filename-display">Welcome</h2>
        <div class="meta" id="counter-display">Select a folder</div>

        <!-- Per-image catalog properties (shown when catalog available) -->
        <div id="catalog-props-box" class="catalog-props-box">
            <div class="props-title">Catalog Properties</div>
            <div id="catalog-props-content"></div>
        </div>

        <label class="section-label">Folder</label>
        <select id="folder-select" class="dropdown" onchange="loadFolder()">
            <option value="" disabled selected>Loading folders...</option>
        </select>
        
        <label class="section-label">Search</label>
        <input type="text" id="search-input" class="search-box" placeholder="Filename..." oninput="filterImages()">

        <label class="section-label">Filter Status</label>
        <select id="filter-select" class="dropdown" onchange="fetchImages()">
            <option value="All">All Images</option>
            <optgroup label="My Classifications">
                <option value="Uncategorized">Uncategorized</option>
                <option value="Classified">Classified (me)</option>
            </optgroup>
            <optgroup label="Multi-User">
                <option value="NoVotes">Unclassified by anyone</option>
                <option value="Disagree">Has Disagreement</option>
                <option value="Agree">Unanimous Agreement</option>
            </optgroup>
            <optgroup label="By Category" id="filter-group-categories"></optgroup>
            <optgroup label="By Tag" id="filter-group-tags">
                <option value="Untagged">No tags</option>
            </optgroup>
        </select>

        <label class="section-label">Filter by User</label>
        <select id="filter-user-select" class="dropdown" onchange="fetchImages()">
            <option value="">My Classifications</option>
        </select>

        <div id="assignment-notice" style="display:none; background:#eff6ff; border:1px solid #bfdbfe; border-radius:6px; padding:8px 12px; margin-bottom:1rem; font-size:0.85rem; color:#1d4ed8;">
            <div>&#128204; Assignment: <b id="assignment-count-display"></b> images</div>
            <label style="display:flex; align-items:center; gap:6px; margin-top:5px; cursor:pointer; color:#374151; font-size:0.82rem;">
                <input type="checkbox" id="show-assigned-only" onchange="toggleAssignedOnly()"> Show assigned images only
            </label>
        </div>

        <!-- Catalog sort / filter (shown only when catalog.fits is available) -->
        <div id="catalog-section" style="display:none">
            <label class="section-label">Sort by Catalog Column</label>
            <div class="catalog-sort-row">
                <select id="sort-col" class="dropdown" onchange="applyAllFilters(true)">
                    <option value="">Default Order</option>
                </select>
                <select id="sort-dir" class="dropdown" onchange="applyAllFilters(true)">
                    <option value="asc">&#8593; Asc</option>
                    <option value="desc">&#8595; Desc</option>
                </select>
            </div>

            <label class="section-label">Filter by Catalog Column</label>
            <select id="filter-col" class="dropdown" onchange="updateFilterRange()">
                <option value="">No filter</option>
            </select>
            <div id="filter-range-row" style="display:none">
                <div class="filter-range-row">
                    <span>Min</span>
                    <input type="number" id="filter-min" class="filter-range-input" placeholder="-&infin;" oninput="applyAllFilters()">
                    <span>Max</span>
                    <input type="number" id="filter-max" class="filter-range-input" placeholder="+&infin;" oninput="applyAllFilters()">
                </div>
            </div>

            <label style="display:flex; align-items:center; gap:6px; font-size:0.85rem; color:var(--text-sub); margin-bottom:1rem; cursor:pointer;">
                <input type="checkbox" id="filter-no-catalog" onchange="applyAllFilters()">
                Show only images not in catalog
            </label>

            <label class="section-label">Mask Expression</label>
            <input type="text" id="mask-expr-input" class="mask-expr-input"
                placeholder="e.g. mag &lt; 20 &amp; snr &gt; 5"
                onkeydown="if(event.key==='Enter') applyMaskExpr()">
            <div class="mask-expr-row">
                <button class="btn-secondary" onclick="applyMaskExpr()"
                    style="flex:1;padding:5px 10px;font-size:0.82rem;">Apply</button>
                <button class="btn-secondary" onclick="clearMaskExpr()"
                    style="white-space:nowrap;padding:5px 10px;font-size:0.82rem;" title="Clear mask">&#10005;</button>
            </div>
            <div id="mask-expr-status" class="mask-expr-status"></div>
        </div>
        <div id="catalog-loading" class="catalog-loading" style="display:none">Loading catalog&#8230;</div>
        <button id="btn-scatter" class="btn-scatter" onclick="openScatterPlot()" style="display:none">Scatter Plot</button>
        <button id="btn-lf" class="btn-lf" onclick="openLF()" style="display:none">Luminosity Function (1/Vmax)</button>
        <button id="btn-sky" class="btn-sky" onclick="toggleSkyPlot()" style="display:none">Sky Position</button>
        <button id="btn-compare" class="btn-compare" onclick="openCompare()" style="display:none">Compare Users</button>

        <label class="section-label">Classification</label>
        <div id="category-container" class="btn-grid"></div>
        
        <!-- Stats Toggle -->
        <button id="btn-stats" class="btn-secondary" onclick="toggleStats()">Show Community Stats</button>
        <div id="stats-box" class="stats-box">
            <div id="stats-content">Loading...</div>
        </div>

        <!-- Tags: shared, user-defined groupings; click a chip to toggle it on the current image -->
        <label class="section-label" style="display:flex; justify-content:space-between; align-items:baseline;">
            <span>Tags <span style="font-weight:400; color:var(--text-sub); font-size:0.78rem;">(shared)</span></span>
            <span id="tag-manage-toggle" onclick="toggleTagManage()" style="font-size:0.78rem; font-weight:500; color:var(--accent); cursor:pointer;">Manage</span>
        </label>
        <div id="tag-row" class="tag-row"><span class="tag-empty">No tags defined yet</span></div>
        <div class="tag-new-row">
            <input type="text" id="tag-new-input" placeholder="New tag&#8230; (Enter also applies it)" maxlength="100"
                onkeydown="if(event.key==='Enter') createTag()">
            <button onclick="createTag()">+ Add</button>
        </div>
        <div id="tag-manage-box" class="tag-manage-box">
            <div>Click <b>&#8595;</b> on a tag to export it as CSV, <b>&#10005;</b> to delete it (creator or admin only). Bulk-apply to the <b id="tag-bulk-count">0</b> images currently shown:</div>
            <div class="row">
                <select id="tag-bulk-select"></select>
                <button onclick="bulkTag('add')" title="Add the selected tag to every image in the current filtered list">Tag all</button>
                <button onclick="bulkTag('remove')" title="Remove the selected tag from every image in the current filtered list">Untag all</button>
            </div>
        </div>

        <label class="section-label">Notes</label>
        <textarea id="notes-input" class="notes-area" placeholder="Private notes..."></textarea>

        <div id="export-section">
            <span class="section-label">Export / Import</span>
            <div class="export-row">
                <button class="btn-export" onclick="exportClassifications('csv')">&#8595; CSV</button>
                <button class="btn-export" onclick="exportClassifications('fits')">&#8595; FITS</button>
                <label class="export-check">
                    <input type="checkbox" id="export-all-users"> All users
                </label>
            </div>
            <div class="export-row" style="margin-top:0.4rem;">
                <button class="btn-export" onclick="openImport()" style="color:#059669; border-color:#a7f3d0;">&#8593; Import CSV / FITS</button>
            </div>
        </div>

        <button id="btn-shuffle" class="btn-secondary" onclick="toggleShuffle()" style="display:none; margin-bottom:0.5rem;">Shuffle</button>
        <button id="btn-autoscroll" class="btn-secondary" onclick="toggleAutoscroll()" style="display:none; margin-bottom:0.5rem;" title="After classifying, jump straight to the next image">Autoscroll</button>
        <div class="nav-row">
            <button class="btn-nav" onclick="navigate(-1)">Previous</button>
            <button class="btn-nav" onclick="navigate(1)">Next</button>
        </div>
    </div>
</div>
<!-- Scatter Plot Modal -->
<div id="scatter-modal">
    <div class="scatter-panel">
        <div class="scatter-header">
            <div style="display:flex; align-items:center;">
                <h3>Scatter Plot</h3>
                <span id="scatter-point-count" class="scatter-point-count" style="display:none;"></span>
            </div>
            <button class="btn-close" onclick="closeScatterPlot()">&#x2715;</button>
        </div>
        <div class="scatter-controls">
            <div class="scatter-ctrl">
                <label>X Axis</label>
                <select id="scatter-x" onchange="renderScatterPlot()"></select>
            </div>
            <div class="scatter-ctrl">
                <label>Y Axis</label>
                <select id="scatter-y" onchange="renderScatterPlot()"></select>
            </div>
            <div class="scatter-ctrl">
                <label>Color by</label>
                <select id="scatter-color" onchange="renderScatterPlot()">
                    <option value="_none">None</option>
                    <option value="_classification">Classification</option>
                </select>
            </div>
            <div class="scatter-ctrl">
                <label>Log scale</label>
                <div style="display:flex; gap:0.75rem; padding-top:4px;">
                    <label class="scatter-ctrl-check"><input type="checkbox" id="scatter-logx" onchange="renderScatterPlot()"> X</label>
                    <label class="scatter-ctrl-check"><input type="checkbox" id="scatter-logy" onchange="renderScatterPlot()"> Y</label>
                </div>
            </div>
        </div>
        <div id="scatter-plot"></div>
    </div>
</div>

<!-- Luminosity Function Modal -->
<div id="lf-modal">
    <div class="lf-panel">
        <div class="scatter-header">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <h3 style="margin:0;">1/V<sub>max</sub> Luminosity Function</h3>
                <span id="lf-meta" class="lf-meta"></span>
            </div>
            <button class="btn-close" onclick="closeLF()">&#x2715;</button>
        </div>
        <div class="lf-controls">
            <div class="lf-ctrl">
                <label>Magnitude column</label>
                <select id="lf-mag" onchange="runLF()"></select>
            </div>
            <div class="lf-ctrl">
                <label>Redshift column</label>
                <select id="lf-z" onchange="runLF()"></select>
            </div>
            <div class="lf-ctrl">
                <label>Parent sample</label>
                <select id="lf-parent" onchange="runLF()"></select>
            </div>
            <div class="lf-ctrl">
                <label>z min</label>
                <input type="number" id="lf-zmin" class="num" step="0.1" value="0" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>z max</label>
                <input type="number" id="lf-zmax" class="num" step="0.1" value="1" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>Area (deg&sup2;)</label>
                <input type="number" id="lf-area" class="num" step="any" value="1" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>Bin width (mag)</label>
                <input type="number" id="lf-binw" class="num" step="0.1" value="0.5" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>H&#8320;</label>
                <input type="number" id="lf-h0" class="num" step="any" value="70" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>&Omega;<sub>m</sub></label>
                <input type="number" id="lf-om0" class="num" step="any" value="0.3" onchange="runLF()">
            </div>
            <div class="lf-ctrl">
                <label>Mask cut A <span id="lf-use-mask" style="color:#9333ea;cursor:pointer;text-transform:none;letter-spacing:0;" onclick="lfUseCurrentMask()" title="Copy the sidebar mask expression">&#8631; current</span></label>
                <input type="text" id="lf-expr-a" class="expr" placeholder="(no cut)" onkeydown="if(event.key==='Enter')runLF()">
            </div>
            <div class="lf-ctrl">
                <label>Mask cut B (compare)</label>
                <input type="text" id="lf-expr-b" class="expr" placeholder="(off)" onkeydown="if(event.key==='Enter')runLF()">
            </div>
            <div class="lf-ctrl">
                <label>Options</label>
                <div style="display:flex; gap:0.6rem; padding-top:3px;">
                    <label class="lf-ctrl-check"><input type="checkbox" id="lf-logy" checked onchange="renderLF()"> log&#966;</label>
                    <label class="lf-ctrl-check"><input type="checkbox" id="lf-err" checked onchange="renderLF()"> errors</label>
                </div>
            </div>
            <div class="lf-ctrl">
                <label>&nbsp;</label>
                <button class="btn-secondary" style="padding:6px 12px;font-size:0.85rem;" onclick="runLF()">Compute</button>
            </div>
            <div class="lf-ctrl">
                <label>&nbsp;</label>
                <button class="btn-secondary" style="padding:6px 12px;font-size:0.85rem;" onclick="lfExportCsv()" title="Download binned phi(M) table">&#8595; CSV</button>
            </div>
        </div>
        <div id="lf-status" class="lf-status"></div>
        <div class="lf-body">
            <div id="lf-plot"></div>
            <div class="lf-side">
                <h4 id="lf-side-title">Click a bin</h4>
                <div id="lf-gal-list" class="lf-gal-list">
                    <div style="color:var(--text-sub);font-size:0.82rem;">Click a histogram bin to list the galaxies that contribute to it. Click a galaxy to jump to its image.</div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Comparison Modal -->
<div id="compare-modal">
    <div class="compare-panel">
        <div class="compare-header">
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <h3>User Comparison</h3>
                <span id="compare-count" class="scatter-point-count" style="display:none;"></span>
            </div>
            <button class="btn-close" onclick="closeCompare()">&#x2715;</button>
        </div>
        <div class="compare-controls">
            <div class="scatter-ctrl">
                <label>Show</label>
                <select id="compare-filter" onchange="renderCompareTable()">
                    <option value="all">All classified</option>
                    <option value="disagree">Disagreements only</option>
                    <option value="agree">Unanimous only</option>
                    <option value="mine">Not yet classified by me</option>
                </select>
            </div>
            <div style="margin-left:auto; font-size:0.8rem; color:#6b7280; align-self:flex-end; padding-bottom:4px;">
                Click a row to navigate to that source
            </div>
        </div>
        <div id="compare-table-wrapper">
            <table id="compare-table">
                <thead id="compare-thead"></thead>
                <tbody id="compare-tbody"></tbody>
            </table>
        </div>
    </div>
</div>

<!-- Import Modal -->
<div id="import-modal">
    <div class="import-panel">
        <div class="import-header">
            <h3>Import Classifications</h3>
            <button class="btn-close" onclick="closeImport()">&#x2715;</button>
        </div>

        <!-- Step 1: upload -->
        <div id="import-step-upload">
            <p style="color:#6b7280; font-size:0.875rem; margin:0 0 1rem;">
                Upload a CSV or FITS file containing <b>ID</b> and <b>category</b> columns.
                Only IDs that exist as images in the current folder will be imported.
                Notes columns are ignored.
            </p>
            <div id="import-dropzone" onclick="document.getElementById('import-file-input').click()">
                <input type="file" id="import-file-input" accept=".csv,.fits,.fit" style="display:none" onchange="handleImportFile()">
                <div id="import-dropzone-inner">
                    <div style="font-size:1.8rem; margin-bottom:0.4rem;">&#128196;</div>
                    Click to select a CSV or FITS file
                </div>
                <div id="import-parsing" style="display:none; font-style:italic;">Parsing file&#8230;</div>
            </div>
            <div id="import-parse-error" style="display:none; color:#dc2626; font-size:0.875rem; margin-top:0.75rem;"></div>
        </div>

        <!-- Step 2: preview + conflict choice -->
        <div id="import-step-preview" style="display:none;">
            <div id="import-summary"></div>
            <div id="import-conflict-section" style="display:none; margin-top:0.75rem;">
                <div class="section-label" style="margin-bottom:0.4rem;">Conflict resolution</div>
                <div class="import-conflict-opts">
                    <label><input type="radio" name="conflict-mode" value="skip" checked> <span><b>Skip</b> — keep my existing classifications</span></label>
                    <label><input type="radio" name="conflict-mode" value="overwrite"> <span><b>Overwrite</b> — replace with imported values</span></label>
                </div>
                <div id="import-preview-rows"></div>
            </div>
            <div style="display:flex; gap:0.75rem; margin-top:1rem;">
                <button onclick="resetImport()" class="btn-secondary" style="flex:1; margin:0;">&#8592; Back</button>
                <button onclick="commitImport()" id="btn-import-confirm" class="btn-nav" style="flex:2;">Import</button>
            </div>
        </div>

        <!-- Step 3: result -->
        <div id="import-step-done" style="display:none;">
            <div id="import-result"></div>
            <button onclick="closeImport()" class="btn-nav" style="width:100%; margin-top:1rem;">Done</button>
        </div>
    </div>
</div>

<div id="toast">Saved</div>

<script>
    let images = [];
    let serverImages = []; // Cache for images fetched from server
    let currentIndex = 0;
    let categories = [];
    let currentFolder = "";
    let showStats = false;
    let catalogData = null; // null = not loaded yet; false = unavailable; object = loaded
    let classificationMap = {}; // image_key → category, fetched per folder
    let compareData = null;
    let importParsedData = null;
    let shuffleMode = false;
    let autoscrollMode = false;
    let currentAssignment = null;
    let showAssignedOnly = false;
    let maskPassIds = null; // null = no mask active; Set<string> = IDs passing the mask expr
    const catColors = {};
    let raCol = null;
    let decCol = null;
    let skyPlotBuilt = false;
    let lfData = null; // last /api/catalog/vmax response
    let allTags = []; // [{id, name, color, count, total, can_delete}] shared across users and folders
    let currentImageTags = new Map(); // tag id → {username, ...} for the current image
    let tagManageMode = false;

    function getCatColor(cat) { return catColors[cat] || '#6b7280'; }

    async function loadCategories(folder) {
        const url = folder ? `/api/config?folder=${encodeURIComponent(folder)}` : '/api/config';
        const configRes = await fetch(url);
        const configData = await configRes.json();
        categories = configData.categories;

        const container = document.getElementById('category-container');
        container.innerHTML = '';
        categories.forEach((cat, index) => {
            const btn = document.createElement('button');
            btn.className = 'btn-cat';
            if (index < 9) {
                btn.innerText = `${index + 1}. ${cat}`;
            } else {
                btn.innerText = cat;
            }
            btn.onclick = () => selectCategory(cat);
            btn.dataset.category = cat;
            container.appendChild(btn);
        });

        const filterGroup = document.getElementById('filter-group-categories');
        filterGroup.innerHTML = '';
        categories.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.innerText = cat;
            filterGroup.appendChild(opt);
        });

        const _palette = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];
        categories.forEach((cat, i) => { catColors[cat] = _palette[i % _palette.length]; });
    }

    async function init() {
        // 1. Load Config (Categories)
        await loadCategories(null);

        // 2. Fetch Folders, Users and Tags
        await Promise.all([fetchFolders(), fetchUsers(), fetchTags()]);

        // 3. Setup Shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
            
            // Navigation
            if (e.key === "ArrowLeft") navigate(-1);
            if (e.key === "ArrowRight") navigate(1);

            // Number Keys 1-9
            const key = parseInt(e.key);
            if (!isNaN(key) && key > 0 && key <= categories.length) {
                // key 1 maps to index 0
                selectCategory(categories[key - 1]);
            }
        });
        
        document.getElementById('notes-input').addEventListener('blur', saveCurrentState);
    }
    
    // --- Stats Logic ---
    function toggleStats() {
        showStats = !showStats;
        const statsBox = document.getElementById('stats-box');
        const btn = document.getElementById('btn-stats');
        
        if (showStats) {
            statsBox.style.display = 'block';
            btn.innerText = "Hide Community Stats";
            fetchStats(); // Fetch immediately
        } else {
            statsBox.style.display = 'none';
            btn.innerText = "Show Community Stats";
        }
    }

    async function fetchStats() {
        if (!showStats || currentIndex === -1) return;
        const key = images[currentIndex];
        const container = document.getElementById('stats-content');
        container.innerHTML = 'Loading...';

        try {
            const res = await fetch(`/api/stats?key=${encodeURIComponent(key)}`);
            const data = await res.json();
            container.innerHTML = '';

            const { counts = {}, votes = [] } = data;

            if (votes.length === 0) {
                container.innerHTML = '<div style="color:#6b7280; text-align:center;">No votes yet</div>';
                return;
            }

            // Summary counts
            for (const [cat, count] of Object.entries(counts)) {
                const row = document.createElement('div');
                row.className = 'stat-row';
                row.innerHTML = `<span>${cat || 'Uncategorized'}</span> <b>${count}</b>`;
                container.appendChild(row);
            }

            // Per-user breakdown
            const divider = document.createElement('div');
            divider.style.cssText = 'border-top:1px dashed #e5e7eb; margin:6px 0 4px; font-size:0.78em; color:#9ca3af; text-transform:uppercase; letter-spacing:0.05em;';
            divider.textContent = 'Per user';
            container.appendChild(divider);
            votes.forEach(v => {
                const row = document.createElement('div');
                row.className = 'stat-row';
                const color = getCatColor(v.category);
                const dot = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${color};margin-right:5px;flex-shrink:0;"></span>`;
                row.innerHTML = `<span style="display:flex;align-items:center;">${dot}${v.username}</span><b>${v.category}</b>`;
                container.appendChild(row);
            });
        } catch(e) {
            container.innerHTML = 'Error fetching stats';
        }
    }

    async function fetchFolders() {
        const select = document.getElementById('folder-select');
        try {
            const res = await fetch('/api/folders');
            const folders = await res.json();
            
            if (folders.length === 0) {
                select.innerHTML = '<option value="" disabled selected>No folders found</option>';
            } else {
                select.innerHTML = '<option value="" disabled selected>Select Folder</option>';
                folders.forEach(f => {
                    const opt = document.createElement('option');
                    opt.value = f;
                    opt.innerText = f;
                    select.appendChild(opt);
                });
            }
        } catch (e) {
            console.error(e);
            select.innerHTML = '<option value="" disabled selected>Error loading folders</option>';
        }
    }

    async function fetchUsers() {
        const select = document.getElementById('filter-user-select');
        try {
            const res = await fetch('/api/users');
            const users = await res.json();
            users.forEach(u => {
                const opt = document.createElement('option');
                opt.value = u.id;
                opt.innerText = u.username;
                select.appendChild(opt);
            });
        } catch (e) {
            console.error('Error fetching users:', e);
        }
    }

    async function loadFolder() {
        currentFolder = document.getElementById('folder-select').value;
        // Reset state
        catalogData = null;
        classificationMap = {};
        compareData = null;
        currentAssignment = null;
        showAssignedOnly = false;
        shuffleMode = false;
        maskPassIds = null;
        currentImageTags = new Map();
        document.getElementById('mask-expr-input').value = '';
        document.getElementById('mask-expr-status').textContent = '';
        document.getElementById('mask-expr-status').className = 'mask-expr-status';
        document.getElementById('mask-expr-input').className = 'mask-expr-input';
        document.getElementById('catalog-section').style.display = 'none';
        document.getElementById('btn-scatter').style.display = 'none';
        document.getElementById('btn-lf').style.display = 'none';
        document.getElementById('btn-sky').style.display = 'none';
        document.getElementById('catalog-loading').style.display = 'block';
        document.getElementById('catalog-props-box').style.display = 'none';
        closeSkyPlot();
        document.getElementById('filter-no-catalog').checked = false;
        document.getElementById('export-section').style.display = 'block';
        document.getElementById('btn-compare').style.display = 'block';
        document.getElementById('btn-shuffle').style.display = 'block';
        document.getElementById('btn-shuffle').textContent = 'Shuffle';
        document.getElementById('btn-shuffle').classList.remove('active');
        // Autoscroll is a workflow preference, so it survives folder switches
        document.getElementById('btn-autoscroll').style.display = 'block';
        renderAutoscrollBtn();
        document.getElementById('assignment-notice').style.display = 'none';
        document.getElementById('show-assigned-only').checked = false;
        // Fetch folder-specific categories (may override defaults), then images/catalog/assignment
        await loadCategories(currentFolder);
        await Promise.all([fetchImages(), fetchCatalog(), fetchMyAssignment(), fetchTags()]);
        document.getElementById('catalog-loading').style.display = 'none';
    }

    async function fetchImages() {
        if (!currentFolder) return;
        const filterVal = document.getElementById('filter-select').value;
        const filterUserId = document.getElementById('filter-user-select').value;
        let url = `/api/images?folder=${encodeURIComponent(currentFolder)}&filter=${encodeURIComponent(filterVal)}`;
        if (filterUserId) url += `&filter_user_id=${encodeURIComponent(filterUserId)}`;
        const res = await fetch(url);
        serverImages = await res.json();
        applyAllFilters();
    }

    async function fetchCatalog() {
        if (!currentFolder) return;
        try {
            const res = await fetch(`/api/catalog?folder=${encodeURIComponent(currentFolder)}`);
            const data = await res.json();
            if (data.available) {
                catalogData = data;
                setupCatalogUI();
            } else {
                catalogData = false;
            }
        } catch(e) {
            console.error('Error fetching catalog:', e);
            catalogData = false;
        }
    }

    function setupCatalogUI() {
        const sortCol = document.getElementById('sort-col');
        const filterCol = document.getElementById('filter-col');
        const scatterX = document.getElementById('scatter-x');
        const scatterY = document.getElementById('scatter-y');
        const scatterColor = document.getElementById('scatter-color');
        const lfMag = document.getElementById('lf-mag');
        const lfZ = document.getElementById('lf-z');

        sortCol.innerHTML = '<option value="">Default Order</option>';
        filterCol.innerHTML = '<option value="">No filter</option>';
        scatterX.innerHTML = '';
        scatterY.innerHTML = '';
        lfMag.innerHTML = '';
        lfZ.innerHTML = '';
        // Color-by: keep the two fixed options then add catalog columns
        scatterColor.innerHTML = '<option value="_none">None</option><option value="_classification">Classification</option>';

        catalogData.columns.forEach((col, i) => {
            if (col === 'ID') return;
            const mkOpt = (val, label) => { const o = document.createElement('option'); o.value = val; o.innerText = label; return o; };
            sortCol.appendChild(mkOpt(col, col));
            filterCol.appendChild(mkOpt(col, col));
            scatterColor.appendChild(mkOpt(col, col));
            scatterX.appendChild(mkOpt(col, col));
            scatterY.appendChild(mkOpt(col, col));
            lfMag.appendChild(mkOpt(col, col));
            lfZ.appendChild(mkOpt(col, col));
        });
        // Pre-select second non-ID column as default Y axis
        if (scatterY.options.length > 1) scatterY.selectedIndex = 1;

        // Auto-detect default LF magnitude (M_UV) and redshift columns
        const magPat = /^(m_?uv|muv|abs_?mag_?uv|m_?1500|m1500)$/i;
        const zPat = /^(z|redshift|z_?phot|zphot|z_?spec|zspec|z_?best|zbest|z_?ml)$/i;
        let magPick = '', zPick = '';
        for (const col of catalogData.columns) {
            if (col === 'ID') continue;
            if (!magPick && magPat.test(col)) magPick = col;
            if (!zPick && zPat.test(col)) zPick = col;
        }
        if (magPick) lfMag.value = magPick;
        else if (lfMag.options.length) lfMag.selectedIndex = 0;
        if (zPick) lfZ.value = zPick;
        else if (lfZ.options.length) lfZ.selectedIndex = 0;

        // Parent-sample selector: all sources, current sidebar filter, or one classification
        const lfParent = document.getElementById('lf-parent');
        lfParent.innerHTML =
            '<option value="_all">All catalog sources</option>' +
            '<option value="_filtered">Current sidebar filter</option>';
        categories.forEach(cat => {
            const o = document.createElement('option');
            o.value = '_class:' + cat;
            o.innerText = 'Classified: ' + cat;
            lfParent.appendChild(o);
        });

        document.getElementById('catalog-section').style.display = 'block';
        document.getElementById('btn-scatter').style.display = 'block';
        document.getElementById('btn-lf').style.display = 'block';

        // Detect RA/Dec columns for sky plot
        raCol = null; decCol = null; skyPlotBuilt = false;
        const raPat = /^(ra|alpha|alpha_j2000|ra_deg|right_ascension)$/i;
        const decPat = /^(dec|delta|delta_j2000|dec_deg|declination)$/i;
        for (const col of catalogData.columns) {
            if (!raCol && raPat.test(col)) raCol = col;
            if (!decCol && decPat.test(col)) decCol = col;
        }
        document.getElementById('btn-sky').style.display = (raCol && decCol) ? 'block' : 'none';
    }

    // Extract filename stem (no extension) to match catalog ID column
    function getImageId(key) {
        const filename = key.split('/').pop();
        return filename.replace(/\\.[^.]+$/, '');
    }

    // Called when the filter column dropdown changes
    function updateFilterRange() {
        const col = document.getElementById('filter-col').value;
        const rangeRow = document.getElementById('filter-range-row');
        rangeRow.style.display = col ? 'block' : 'none';
        document.getElementById('filter-min').value = '';
        document.getElementById('filter-max').value = '';
        applyAllFilters();
    }

    // Unified filter: applies catalog filter → catalog sort → search term
    // Pass resetPosition=true (sorting controls) to always jump to index 0.
    function applyAllFilters(resetPosition = false) {
        const currentKey = !resetPosition && currentIndex >= 0 && currentIndex < images.length ? images[currentIndex] : null;

        let result = [...serverImages];

        if (catalogData) {
            // 1. Catalog column range filter
            const filterCol = document.getElementById('filter-col').value;
            if (filterCol) {
                const fMin = parseFloat(document.getElementById('filter-min').value);
                const fMax = parseFloat(document.getElementById('filter-max').value);
                result = result.filter(key => {
                    const row = catalogData.rows[getImageId(key)];
                    if (!row) return true; // keep images not in catalog
                    const val = row[filterCol];
                    if (val === null || val === undefined) return true;
                    if (!isNaN(fMin) && val < fMin) return false;
                    if (!isNaN(fMax) && val > fMax) return false;
                    return true;
                });
            }

            // 2. Not-in-catalog filter
            if (document.getElementById('filter-no-catalog').checked) {
                result = result.filter(key => !catalogData.rows[getImageId(key)]);
            }

            // 3. Mask expression filter (server-computed; IDs not in catalog are kept)
            if (maskPassIds !== null) {
                result = result.filter(key => {
                    const id = getImageId(key);
                    return !catalogData.rows[id] || maskPassIds.has(id);
                });
            }

            // 4. Catalog column sort
            const sortCol = document.getElementById('sort-col').value;
            if (sortCol) {
                const asc = document.getElementById('sort-dir').value === 'asc';
                result.sort((a, b) => {
                    const rowA = catalogData.rows[getImageId(a)];
                    const rowB = catalogData.rows[getImageId(b)];
                    const vA = rowA ? rowA[sortCol] : null;
                    const vB = rowB ? rowB[sortCol] : null;
                    if (vA === null && vB === null) return 0;
                    if (vA === null) return 1;  // nulls go to end
                    if (vB === null) return -1;
                    if (asc) return vA < vB ? -1 : vA > vB ? 1 : 0;
                    return vA > vB ? -1 : vA < vB ? 1 : 0;
                });
            }
        }

        // 3. Client-side search filter
        const term = document.getElementById('search-input').value.toLowerCase();
        result = term ? result.filter(k => k.toLowerCase().includes(term)) : result;

        // 4. Assignment filter
        if (showAssignedOnly && currentAssignment) {
            const assignedSet = new Set(currentAssignment.keys);
            result = result.filter(k => assignedSet.has(k));
        }

        // 5. Shuffle
        if (shuffleMode) {
            for (let i = result.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [result[i], result[j]] = [result[j], result[i]];
            }
        }

        images = result;
        document.getElementById('tag-bulk-count').textContent = images.length;

        if (images.length > 0) {
            // Stay on the current image if it survived the filter; otherwise go to 0
            const preservedIdx = currentKey ? images.indexOf(currentKey) : -1;
            loadStateForImage(preservedIdx >= 0 ? preservedIdx : 0);
        } else {
            document.getElementById('filename-display').innerText = "No images found";
            document.getElementById('main-image').src = "";
            document.getElementById('counter-display').innerText = "0 / 0";
            currentIndex = -1;
            document.getElementById('notes-input').value = '';
            document.getElementById('catalog-props-box').style.display = 'none';
            currentImageTags = new Map();
            renderTagChips();
            if (showStats) document.getElementById('stats-content').innerHTML = '';
        }
    }

    // Keep filterImages() as an alias so the search box oninput still works
    function filterImages() { applyAllFilters(); }

    async function applyMaskExpr() {
        const input = document.getElementById('mask-expr-input');
        const statusEl = document.getElementById('mask-expr-status');
        const expr = input.value.trim();
        if (!expr) {
            maskPassIds = null;
            statusEl.textContent = '';
            statusEl.className = 'mask-expr-status';
            input.className = 'mask-expr-input';
            applyAllFilters();
            return;
        }
        statusEl.className = 'mask-expr-status';
        statusEl.textContent = 'Applying…';
        input.className = 'mask-expr-input';
        try {
            const res = await fetch('/api/catalog/mask', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({folder: currentFolder, expr})
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                statusEl.className = 'mask-expr-status err';
                statusEl.textContent = 'Error: ' + (data.error || res.statusText);
                input.className = 'mask-expr-input error';
                maskPassIds = null;
            } else {
                maskPassIds = new Set(data.ids);
                statusEl.className = 'mask-expr-status';
                statusEl.textContent = data.count.toLocaleString() + ' / ' + data.total.toLocaleString() + ' rows pass';
                applyAllFilters();
            }
        } catch(e) {
            statusEl.className = 'mask-expr-status err';
            statusEl.textContent = 'Error: ' + e.message;
            input.className = 'mask-expr-input error';
            maskPassIds = null;
        }
    }

    function clearMaskExpr() {
        const input = document.getElementById('mask-expr-input');
        const statusEl = document.getElementById('mask-expr-status');
        input.value = '';
        input.className = 'mask-expr-input';
        statusEl.textContent = '';
        statusEl.className = 'mask-expr-status';
        maskPassIds = null;
        applyAllFilters();
    }

    // Show the catalog row for the current image
    function updateCatalogProps(key) {
        const box = document.getElementById('catalog-props-box');
        const content = document.getElementById('catalog-props-content');
        if (!catalogData || !catalogData.rows) { box.style.display = 'none'; return; }
        const row = catalogData.rows[getImageId(key)];
        if (!row) { box.style.display = 'none'; return; }
        content.innerHTML = '';
        for (const [col, val] of Object.entries(row)) {
            const div = document.createElement('div');
            div.className = 'prop-row';
            let display = val;
            if (typeof val === 'number') {
                display = Number.isInteger(val) ? val : val.toPrecision(5);
            }
            div.innerHTML = `<span class="prop-key">${col}</span><span class="prop-val">${display ?? '—'}</span>`;
            content.appendChild(div);
        }
        box.style.display = 'block';
    }

    // --- Tags (shared, user-defined groupings) ---

    function showToast(msg, ms = 1500) {
        const t = document.getElementById('toast');
        t.textContent = msg;
        t.style.opacity = '1';
        setTimeout(() => {
            t.style.opacity = '0';
            setTimeout(() => { t.textContent = 'Saved'; }, 300);
        }, ms);
    }

    async function fetchTags() {
        try {
            const url = currentFolder ? `/api/tags?folder=${encodeURIComponent(currentFolder)}` : '/api/tags';
            const res = await fetch(url);
            const data = await res.json();
            allTags = data.tags || [];
        } catch(e) {
            console.error('Error fetching tags:', e);
            allTags = [];
        }
        renderTagFilterOptions();
        renderTagChips();
    }

    function renderTagFilterOptions() {
        // "By Tag" optgroup in the status filter, with per-folder counts
        const sel = document.getElementById('filter-select');
        const group = document.getElementById('filter-group-tags');
        const prev = sel.value;
        group.innerHTML = '';
        const none = document.createElement('option');
        none.value = 'Untagged';
        none.textContent = 'No tags';
        group.appendChild(none);
        allTags.forEach(t => {
            const opt = document.createElement('option');
            opt.value = `tag:${t.id}`;
            opt.textContent = `${t.name} (${t.count})`;
            group.appendChild(opt);
        });
        sel.value = [...sel.options].some(o => o.value === prev) ? prev : 'All';

        // Bulk-apply dropdown in the manage box
        const bulk = document.getElementById('tag-bulk-select');
        const prevBulk = bulk.value;
        bulk.innerHTML = '';
        allTags.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t.id;
            opt.textContent = t.name;
            bulk.appendChild(opt);
        });
        if ([...bulk.options].some(o => o.value === prevBulk)) bulk.value = prevBulk;
    }

    function renderTagChips() {
        const row = document.getElementById('tag-row');
        row.innerHTML = '';
        if (allTags.length === 0) {
            row.innerHTML = '<span class="tag-empty">No tags defined yet</span>';
            return;
        }
        const hasImage = currentIndex >= 0 && currentIndex < images.length;
        allTags.forEach(t => {
            const info = currentImageTags.get(t.id);
            const active = !!info;
            const chip = document.createElement('span');
            chip.className = 'tag-chip' + (active ? ' active' : '') + (hasImage ? '' : ' disabled');
            chip.style.borderColor = t.color;
            chip.style.color = active ? 'white' : t.color;
            chip.style.background = active ? t.color : 'white';
            chip.title = active
                ? `Added by ${info.username || 'unknown'} — click to remove`
                : (hasImage ? 'Click to add this tag' : `${t.total} image(s) tagged overall`);
            const name = document.createElement('span');
            name.textContent = t.name;
            chip.appendChild(name);
            const cnt = document.createElement('span');
            cnt.className = 'tag-count';
            cnt.textContent = t.count;
            cnt.title = `${t.count} in this folder`;
            chip.appendChild(cnt);
            if (hasImage) chip.onclick = () => toggleTag(t.id);
            if (tagManageMode) {
                const exp = document.createElement('span');
                exp.className = 'tag-icon';
                exp.innerHTML = '&#8595;';
                exp.title = `Export "${t.name}" (this folder) as CSV`;
                exp.onclick = (e) => { e.stopPropagation(); exportTag(t.id); };
                chip.appendChild(exp);
                if (t.can_delete) {
                    const x = document.createElement('span');
                    x.className = 'tag-icon';
                    x.innerHTML = '&#10005;';
                    x.title = `Delete tag "${t.name}"`;
                    x.onclick = (e) => { e.stopPropagation(); deleteTag(t.id); };
                    chip.appendChild(x);
                }
            }
            row.appendChild(chip);
        });
    }

    async function fetchImageTags(key) {
        try {
            const res = await fetch(`/api/image_tags?key=${encodeURIComponent(key)}`);
            const data = await res.json();
            // Discard if the user has already moved to another image
            if (currentIndex < 0 || images[currentIndex] !== key) return;
            currentImageTags = new Map((data.tags || []).map(t => [t.id, t]));
        } catch(e) {
            console.error('Error fetching image tags:', e);
            currentImageTags = new Map();
        }
        renderTagChips();
    }

    async function toggleTag(tagId) {
        if (currentIndex < 0 || currentIndex >= images.length) return;
        const key = images[currentIndex];
        const action = currentImageTags.has(tagId) ? 'remove' : 'add';
        try {
            const res = await fetch(`/api/image_tags/${action}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ tag_id: tagId, keys: [key] })
            });
            const data = await res.json();
            if (!res.ok) { alert(data.error || 'Failed to update tag'); return; }
            if (currentIndex < 0 || images[currentIndex] !== key) return; // navigated away meanwhile
            const tag = allTags.find(t => t.id === tagId);
            if (action === 'add') {
                currentImageTags.set(tagId, { id: tagId, username: data.username });
                if (tag && data.changed) tag.count += 1;
            } else {
                currentImageTags.delete(tagId);
                if (tag && data.changed) tag.count = Math.max(0, tag.count - 1);
            }
            renderTagChips();
            renderTagFilterOptions();
        } catch(e) {
            console.error('Error toggling tag:', e);
        }
    }

    async function createTag() {
        const input = document.getElementById('tag-new-input');
        const name = input.value.trim();
        if (!name) return;
        try {
            const res = await fetch('/api/tags', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ name, folder: currentFolder })
            });
            const data = await res.json();
            if (!res.ok) { alert(data.error || 'Could not create tag'); return; }
            input.value = '';
            // Refetch rather than patch locally: picks up tags other users created or deleted meanwhile
            await fetchTags();
            // Typing a tag while viewing an image almost always means "tag this one", so apply it
            if (currentIndex >= 0 && currentIndex < images.length && !currentImageTags.has(data.tag.id)) {
                await toggleTag(data.tag.id);
            }
        } catch(e) {
            console.error('Error creating tag:', e);
        }
    }

    async function deleteTag(tagId) {
        const tag = allTags.find(t => t.id === tagId);
        if (!tag) return;
        if (!confirm(`Delete tag "${tag.name}"? It will be removed from all ${tag.total} tagged image(s), in every folder.`)) return;
        try {
            const res = await fetch(`/api/tags/${tagId}`, { method: 'DELETE' });
            const data = await res.json();
            if (!res.ok) { alert(data.error || 'Could not delete tag'); return; }
            allTags = allTags.filter(t => t.id !== tagId);
            currentImageTags.delete(tagId);
            const wasFiltering = document.getElementById('filter-select').value === `tag:${tagId}`;
            renderTagFilterOptions();
            renderTagChips();
            if (wasFiltering) fetchImages(); // filter fell back to "All"
        } catch(e) {
            console.error('Error deleting tag:', e);
        }
    }

    function toggleTagManage() {
        tagManageMode = !tagManageMode;
        document.getElementById('tag-manage-box').style.display = tagManageMode ? 'block' : 'none';
        document.getElementById('tag-manage-toggle').textContent = tagManageMode ? 'Done' : 'Manage';
        document.getElementById('tag-bulk-count').textContent = images.length;
        renderTagChips();
    }

    async function bulkTag(action) {
        const tagId = parseInt(document.getElementById('tag-bulk-select').value);
        const tag = allTags.find(t => t.id === tagId);
        if (!tag) { alert('No tag selected'); return; }
        if (images.length === 0) return;
        const verb = action === 'add' ? 'Add' : 'Remove';
        const prep = action === 'add' ? 'to' : 'from';
        if (!confirm(`${verb} tag "${tag.name}" ${prep} all ${images.length} images currently shown?`)) return;
        try {
            const res = await fetch(`/api/image_tags/${action}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ tag_id: tagId, keys: images })
            });
            const data = await res.json();
            if (!res.ok) { alert(data.error || 'Bulk tag failed'); return; }
            showToast(`${action === 'add' ? 'Tagged' : 'Untagged'} ${data.changed} image(s)`);
            await fetchTags();
            if (currentIndex >= 0 && currentIndex < images.length) fetchImageTags(images[currentIndex]);
        } catch(e) {
            console.error('Error bulk tagging:', e);
        }
    }

    function exportTag(tagId) {
        if (!currentFolder) { alert('Select a folder first'); return; }
        window.location.href = `/api/tags/${tagId}/export?folder=${encodeURIComponent(currentFolder)}`;
    }

    // --- Scatter Plot ---

    async function fetchClassifications() {
        if (!currentFolder) return;
        try {
            const res = await fetch(`/api/classifications?folder=${encodeURIComponent(currentFolder)}`);
            classificationMap = await res.json();
        } catch(e) {
            console.error('Error fetching classifications:', e);
        }
    }

    async function openScatterPlot() {
        document.getElementById('scatter-modal').style.display = 'flex';
        // Refresh classifications each time (user may have classified more since folder load)
        await fetchClassifications();
        renderScatterPlot();
    }

    function closeScatterPlot() {
        document.getElementById('scatter-modal').style.display = 'none';
    }

    // Close modal on backdrop click
    document.getElementById('scatter-modal').addEventListener('click', e => {
        if (e.target === document.getElementById('scatter-modal')) closeScatterPlot();
    });

    // --- Luminosity Function (1/Vmax) ---

    const LF_COLORS = ['#9333ea', '#0891b2']; // A, B

    async function openLF() {
        document.getElementById('lf-modal').style.display = 'flex';
        // Prefill cut A from the active sidebar mask the first time it's empty
        const exprA = document.getElementById('lf-expr-a');
        if (!exprA.value) exprA.value = document.getElementById('mask-expr-input').value.trim();
        // Refresh classifications so a "Classified: …" parent sample is up to date
        await fetchClassifications();
        runLF();
    }

    function closeLF() { document.getElementById('lf-modal').style.display = 'none'; }

    document.getElementById('lf-modal').addEventListener('click', e => {
        if (e.target === document.getElementById('lf-modal')) closeLF();
    });

    function lfUseCurrentMask() {
        document.getElementById('lf-expr-a').value = document.getElementById('mask-expr-input').value.trim();
        runLF();
    }

    async function runLF() {
        if (!catalogData || document.getElementById('lf-modal').style.display !== 'flex') return;
        const statusEl = document.getElementById('lf-status');
        const exprA = document.getElementById('lf-expr-a').value.trim();
        const exprB = document.getElementById('lf-expr-b').value.trim();
        const exprs = [exprA];
        if (exprB) exprs.push(exprB);

        // Parent-sample restriction: null = all catalog sources; otherwise a list of
        // catalog IDs the LF is limited to (current sidebar filter, or one classification).
        const parent = document.getElementById('lf-parent').value;
        let restrictIds = null;
        if (parent === '_filtered') {
            restrictIds = images.map(getImageId);
        } else if (parent.startsWith('_class:')) {
            const wantCat = parent.slice('_class:'.length);
            restrictIds = Object.keys(classificationMap)
                .filter(k => classificationMap[k] === wantCat)
                .map(getImageId);
        }

        const payload = {
            folder: currentFolder,
            mag_col: document.getElementById('lf-mag').value,
            z_col: document.getElementById('lf-z').value,
            z_min: parseFloat(document.getElementById('lf-zmin').value),
            z_max: parseFloat(document.getElementById('lf-zmax').value),
            area_deg2: parseFloat(document.getElementById('lf-area').value),
            bin_width: parseFloat(document.getElementById('lf-binw').value),
            H0: parseFloat(document.getElementById('lf-h0').value),
            Om0: parseFloat(document.getElementById('lf-om0').value),
            exprs,
            restrict_ids: restrictIds,
        };

        statusEl.className = 'lf-status';
        statusEl.textContent = 'Computing…';
        try {
            const res = await fetch('/api/catalog/vmax', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            const data = await res.json();
            if (!res.ok || data.error) {
                statusEl.className = 'lf-status err';
                statusEl.textContent = 'Error: ' + (data.error || res.statusText);
                lfData = null;
                return;
            }
            lfData = data;
            const vmaxExp = data.vmax.toExponential(3);
            document.getElementById('lf-meta').innerHTML =
                `V<sub>max</sub> = ${vmaxExp} Mpc³ &nbsp;·&nbsp; shell ${data.volume_mpc3.toExponential(2)} Mpc³ &nbsp;·&nbsp; sky frac ${data.sky_frac.toExponential(2)}`;
            const counts = data.series.map((s, i) => `${String.fromCharCode(65 + i)}: ${s.count.toLocaleString()} gal`).join(' · ');
            statusEl.className = 'lf-status';
            statusEl.textContent = counts;
            // Reset the side panel since bins changed
            document.getElementById('lf-side-title').textContent = 'Click a bin';
            document.getElementById('lf-gal-list').innerHTML =
                '<div style="color:var(--text-sub);font-size:0.82rem;">Click a histogram bin to list contributing galaxies.</div>';
            renderLF();
        } catch (e) {
            statusEl.className = 'lf-status err';
            statusEl.textContent = 'Error: ' + e.message;
            lfData = null;
        }
    }

    function renderLF() {
        if (!lfData) return;
        const logY = document.getElementById('lf-logy').checked;
        const showErr = document.getElementById('lf-err').checked;
        const magCol = document.getElementById('lf-mag').value;

        const traces = [];
        let xLo = Infinity, xHi = -Infinity;
        lfData.series.forEach((s, si) => {
            if (!s.bins.length) return;
            const color = LF_COLORS[si % LF_COLORS.length];
            s.bins.forEach(b => { if (b.lo < xLo) xLo = b.lo; if (b.hi > xHi) xHi = b.hi; });
            const xs = s.bins.map(b => b.center);
            const ys = s.bins.map(b => b.phi);
            const errs = s.bins.map(b => b.phi_err);
            const name = s.expr ? `${String.fromCharCode(65 + si)}: ${s.expr}` : `${String.fromCharCode(65 + si)}: all`;
            traces.push({
                type: 'scatter', mode: 'lines+markers', name,
                x: xs, y: ys,
                customdata: s.bins.map(b => [si, b.count, b.lo, b.hi]),
                error_y: showErr ? { type: 'data', array: errs, visible: true, thickness: 1, width: 3, color } : { visible: false },
                line: { shape: 'hv', color, width: 2 },
                marker: { size: 8, color },
                hovertemplate: `${magCol}=%{x:.2f}<br>φ=%{y:.3e}<br>N=%{customdata[1]}<extra>${name}</extra>`,
            });
        });

        // Force the magnitude axis to run high → low (left to right).
        const xPad = isFinite(xLo) ? (xHi - xLo) * 0.04 + 0.1 : 1;
        const xRange = isFinite(xLo) ? [xHi + xPad, xLo - xPad] : undefined;

        const layout = {
            xaxis: { title: magCol + ' (mag)', range: xRange, autorange: xRange ? false : 'reversed', showgrid: true, gridcolor: '#e5e7eb', zeroline: false },
            yaxis: { title: 'φ  (Mpc⁻³ mag⁻¹)', type: logY ? 'log' : 'linear', showgrid: true, gridcolor: '#e5e7eb', zeroline: false },
            margin: { l: 75, r: 20, t: 15, b: 55 },
            hovermode: 'closest',
            paper_bgcolor: 'white',
            plot_bgcolor: '#f9fafb',
            legend: { orientation: 'h', x: 0, y: 1.05, xanchor: 'left', yanchor: 'bottom' },
            font: { family: '-apple-system, sans-serif', size: 12 },
        };
        const config = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['toImage'] };

        const plotDiv = document.getElementById('lf-plot');
        Plotly.react(plotDiv, traces, layout, config);

        // Re-binding on every react() would stack handlers; clear first.
        if (plotDiv.removeAllListeners) plotDiv.removeAllListeners('plotly_click');
        plotDiv.on('plotly_click', evt => {
            if (!evt.points.length) return;
            const cd = evt.points[0].customdata; // [seriesIdx, count, lo, hi]
            showLFBinGalaxies(cd[0], evt.points[0].x, cd[2], cd[3]);
        });
    }

    function showLFBinGalaxies(seriesIdx, center, lo, hi) {
        if (!lfData) return;
        const gals = lfData.galaxies[seriesIdx] || [];
        const series = lfData.series[seriesIdx];
        // Find the bin index whose center matches the clicked point
        let binIdx = -1;
        series.bins.forEach((b, i) => { if (Math.abs(b.center - center) < 1e-6) binIdx = i; });
        const members = gals.filter(g => g.bin === binIdx);

        const label = String.fromCharCode(65 + seriesIdx);
        document.getElementById('lf-side-title').textContent =
            `Cut ${label}: ${lo.toFixed(2)} ≤ ${document.getElementById('lf-mag').value} < ${hi.toFixed(2)} — ${members.length} gal`;

        const list = document.getElementById('lf-gal-list');
        if (!members.length) {
            list.innerHTML = '<div style="color:var(--text-sub);font-size:0.82rem;">No galaxies in this bin.</div>';
            return;
        }
        list.innerHTML = '';
        members.forEach(g => {
            const inImages = images.some(k => getImageId(k) === g.id);
            const row = document.createElement('div');
            row.className = 'lf-gal-row';
            row.title = inImages ? 'Jump to this image' : 'Not in the current image list';
            if (!inImages) row.style.opacity = '0.5';
            row.innerHTML = `<span class="gid">${g.id}</span><span class="gval">${g.mag.toFixed(2)} · z=${g.z.toFixed(2)}</span>`;
            if (inImages) row.onclick = () => navigateToGalaxyId(g.id);
            list.appendChild(row);
        });
    }

    function navigateToGalaxyId(id) {
        const key = images.find(k => getImageId(k) === id);
        if (!key) return;
        const idx = images.indexOf(key);
        if (idx !== -1) { loadStateForImage(idx); closeLF(); }
    }

    function lfExportCsv() {
        if (!lfData) return;
        let csv = 'cut,mag_center,mag_lo,mag_hi,count,phi,phi_err\\n';
        lfData.series.forEach((s, si) => {
            const label = String.fromCharCode(65 + si);
            s.bins.forEach(b => {
                csv += `${label},${b.center},${b.lo},${b.hi},${b.count},${b.phi},${b.phi_err}\\n`;
            });
        });
        const blob = new Blob([csv], { type: 'text/csv' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `lumfunc_${currentFolder}.csv`;
        a.click();
        URL.revokeObjectURL(a.href);
    }

    // --- Sky Position Plot ---

    function toggleSkyPlot() {
        const overlay = document.getElementById('sky-overlay');
        if (overlay.classList.contains('open')) {
            closeSkyPlot();
        } else {
            openSkyPlot();
        }
    }

    function openSkyPlot() {
        const overlay = document.getElementById('sky-overlay');
        overlay.classList.add('open');
        document.getElementById('btn-sky').classList.add('active');
        if (!skyPlotBuilt) {
            buildSkyPlot();
        } else {
            updateSkyHighlight();
        }
    }

    function closeSkyPlot() {
        document.getElementById('sky-overlay').classList.remove('open');
        document.getElementById('btn-sky').classList.remove('active');
    }

    function buildSkyPlot() {
        if (!catalogData || !raCol || !decCol) return;

        // Build all-sources trace
        const allRa = [], allDec = [], allKeys = [], allText = [];
        for (const [id, row] of Object.entries(catalogData.rows)) {
            const ra = row[raCol], dec = row[decCol];
            if (ra === null || ra === undefined || dec === null || dec === undefined) continue;
            allRa.push(ra); allDec.push(dec); allKeys.push(id);
            allText.push(`${id}<br>${raCol}: ${Number(ra).toFixed(5)}<br>${decCol}: ${Number(dec).toFixed(5)}`);
        }

        // Cosine correction: at declination δ, 1° RA = cos(δ)° of arc.
        // scaleratio = 1/cos(δ_mean) makes 1° Dec span the same screen height as cos(δ)° RA spans width.
        const meanDec = allDec.length ? allDec.reduce((a, b) => a + b, 0) / allDec.length : 0;
        const cosDec = Math.cos(meanDec * Math.PI / 180);
        const skyScaleRatio = cosDec > 0.001 ? 1 / cosDec : 1;

        const currentKey = currentIndex >= 0 && currentIndex < images.length ? images[currentIndex] : null;
        const currentId = currentKey ? getImageId(currentKey) : null;
        const curRow = currentId && catalogData.rows[currentId] ? catalogData.rows[currentId] : null;

        const traces = [
            {
                type: 'scatter', mode: 'markers', name: 'All sources',
                x: allRa, y: allDec,
                customdata: allKeys,
                text: allText,
                hovertemplate: '%{text}<extra></extra>',
                marker: { size: 5, color: '#94a3b8', opacity: 0.7 }
            },
            {
                type: 'scatter', mode: 'markers', name: 'Current',
                x: curRow ? [curRow[raCol]] : [],
                y: curRow ? [curRow[decCol]] : [],
                customdata: curRow ? [currentId] : [],
                hovertemplate: curRow ? `${currentId}<br>${raCol}: ${Number(curRow[raCol]).toFixed(5)}<br>${decCol}: ${Number(curRow[decCol]).toFixed(5)}<extra>current</extra>` : '<extra></extra>',
                marker: { size: 14, color: '#ef4444', symbol: 'star', opacity: 1, line: { width: 1, color: '#7f1d1d' } }
            }
        ];

        const layout = {
            xaxis: { title: raCol + ' (deg)', autorange: 'reversed', showgrid: true, gridcolor: '#e5e7eb', zeroline: false },
            yaxis: { title: decCol + ' (deg)', showgrid: true, gridcolor: '#e5e7eb', zeroline: false, scaleanchor: 'x', scaleratio: skyScaleRatio },
            margin: { l: 55, r: 15, t: 10, b: 50 },
            hovermode: 'closest',
            paper_bgcolor: '#f9fafb',
            plot_bgcolor: '#f9fafb',
            legend: { orientation: 'h', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom' },
            font: { family: '-apple-system, sans-serif', size: 11 }
        };
        const config = { responsive: true, displayModeBar: false };

        const plotDiv = document.getElementById('sky-plot-div');
        Plotly.newPlot(plotDiv, traces, layout, config);
        skyPlotBuilt = true;

        plotDiv.on('plotly_click', function(data) {
            if (!data.points.length) return;
            const clickedId = data.points[0].customdata;
            const key = images.find(k => getImageId(k) === clickedId);
            if (!key) return;
            const idx = images.indexOf(key);
            if (idx !== -1) loadStateForImage(idx);
        });
    }

    function updateSkyHighlight() {
        if (!skyPlotBuilt || !raCol || !decCol) return;
        const plotDiv = document.getElementById('sky-plot-div');
        const currentKey = currentIndex >= 0 && currentIndex < images.length ? images[currentIndex] : null;
        const currentId = currentKey ? getImageId(currentKey) : null;
        const curRow = currentId && catalogData && catalogData.rows[currentId] ? catalogData.rows[currentId] : null;

        Plotly.restyle(plotDiv, {
            x: [curRow ? [curRow[raCol]] : []],
            y: [curRow ? [curRow[decCol]] : []],
            customdata: [curRow ? [currentId] : []],
            hovertemplate: [curRow
                ? `${currentId}<br>${raCol}: ${Number(curRow[raCol]).toFixed(5)}<br>${decCol}: ${Number(curRow[decCol]).toFixed(5)}<extra>current</extra>`
                : '<extra></extra>']
        }, [1]);
    }

    // --- Compare Users ---

    async function openCompare() {
        document.getElementById('compare-modal').style.display = 'flex';
        if (!compareData) {
            document.getElementById('compare-tbody').innerHTML =
                '<tr><td colspan="99" style="text-align:center;padding:2rem;color:#6b7280;">Loading&#8230;</td></tr>';
            await fetchCompare();
        }
        renderCompareTable();
    }

    function closeCompare() {
        document.getElementById('compare-modal').style.display = 'none';
    }

    document.getElementById('compare-modal').addEventListener('click', e => {
        if (e.target === document.getElementById('compare-modal')) closeCompare();
    });

    async function fetchCompare() {
        try {
            const res = await fetch(`/api/compare?folder=${encodeURIComponent(currentFolder)}`);
            compareData = await res.json();
        } catch(e) {
            console.error('Error fetching compare data:', e);
            compareData = { users: [], classifications: {} };
        }
    }

    function renderCompareTable() {
        if (!compareData) return;
        const filterVal = document.getElementById('compare-filter').value;
        const { users, classifications } = compareData;
        const myId = '{{ current_user.id }}';

        let keys = Object.keys(classifications);

        if (filterVal === 'disagree') {
            keys = keys.filter(k => {
                const votes = Object.values(classifications[k]);
                return votes.length >= 2 && new Set(votes).size > 1;
            });
        } else if (filterVal === 'agree') {
            keys = keys.filter(k => {
                const votes = Object.values(classifications[k]);
                return votes.length >= 2 && new Set(votes).size === 1;
            });
        } else if (filterVal === 'mine') {
            keys = keys.filter(k => !classifications[k][myId]);
        }

        keys.sort();

        const countEl = document.getElementById('compare-count');
        countEl.textContent = `${keys.length} source${keys.length !== 1 ? 's' : ''}`;
        countEl.style.display = 'inline';

        // Build header
        const thead = document.getElementById('compare-thead');
        thead.innerHTML = '';
        const headerRow = thead.insertRow();
        const thKey = document.createElement('th');
        thKey.textContent = 'Source';
        headerRow.appendChild(thKey);
        users.forEach(u => {
            const th = document.createElement('th');
            th.textContent = u.username + (u.id === myId ? ' (me)' : '');
            if (u.id === myId) th.style.fontStyle = 'italic';
            headerRow.appendChild(th);
        });
        const thStatus = document.createElement('th');
        thStatus.textContent = 'Status';
        headerRow.appendChild(thStatus);

        // Build body
        const tbody = document.getElementById('compare-tbody');
        tbody.innerHTML = '';

        if (keys.length === 0) {
            const tr = tbody.insertRow();
            const td = tr.insertCell();
            td.colSpan = users.length + 2;
            td.style.cssText = 'text-align:center;padding:2rem;color:#6b7280;';
            td.textContent = 'No sources match this filter.';
            return;
        }

        keys.forEach(key => {
            const tr = tbody.insertRow();
            const votes = classifications[key];
            const stem = key.split('/').pop().replace(/\.[^.]+$/, '');

            const tdKey = tr.insertCell();
            tdKey.textContent = stem;
            tdKey.title = key;

            users.forEach(u => {
                const td = tr.insertCell();
                const cat = votes[u.id];
                if (cat) {
                    const span = document.createElement('span');
                    span.className = 'compare-cat-cell';
                    span.textContent = cat;
                    const color = getCatColor(cat);
                    span.style.cssText = `background:${color}22;color:${color};border:1px solid ${color}55;`;
                    td.appendChild(span);
                } else {
                    td.textContent = '—';
                    td.style.color = '#d1d5db';
                }
            });

            // Status cell
            const tdStatus = tr.insertCell();
            const allVotes = Object.values(votes);
            const uniqueCats = new Set(allVotes);
            if (allVotes.length < 2) {
                tdStatus.innerHTML = '<span style="color:#9ca3af;">&#8230;</span>';
                tdStatus.title = 'Fewer than 2 users have classified';
            } else if (uniqueCats.size === 1) {
                tdStatus.innerHTML = '<span class="compare-agree">&#10003;</span>';
                tdStatus.title = 'All agree';
            } else {
                tdStatus.innerHTML = '<span class="compare-disagree">&#10007;</span>';
                tdStatus.title = 'Disagreement';
            }

            // Click to navigate (reset filter if image not in current list)
            tr.addEventListener('click', async () => {
                let idx = images.indexOf(key);
                if (idx === -1) {
                    document.getElementById('filter-select').value = 'All';
                    await fetchImages();
                    idx = images.indexOf(key);
                }
                if (idx !== -1) {
                    closeCompare();
                    loadStateForImage(idx);
                }
            });
        });
    }

    function renderScatterPlot() {
        if (!catalogData) return;
        const xCol = document.getElementById('scatter-x').value;
        const yCol = document.getElementById('scatter-y').value;
        const colorBy = document.getElementById('scatter-color').value;
        const logX = document.getElementById('scatter-logx').checked;
        const logY = document.getElementById('scatter-logy').checked;
        if (!xCol || !yCol) return;

        // Build a point for every image that has catalog data and valid X/Y values.
        // Use `images` so the scatter plot respects the active catalogue range filter.
        const points = [];
        images.forEach(key => {
            const row = catalogData.rows[getImageId(key)];
            if (!row) return;
            const x = row[xCol], y = row[yCol];
            if (x === null || x === undefined || y === null || y === undefined) return;
            points.push({ key, x, y, cat: classificationMap[key] || 'Unclassified', row });
        });

        const currentKey = currentIndex >= 0 && currentIndex < images.length ? images[currentIndex] : null;

        const countEl = document.getElementById('scatter-point-count');
        countEl.textContent = `${points.length} point${points.length !== 1 ? 's' : ''}`;
        countEl.style.display = 'inline';

        const layout = {
            xaxis: { title: xCol, type: logX ? 'log' : 'linear', autorange: true },
            yaxis: { title: yCol, type: logY ? 'log' : 'linear', autorange: true },
            margin: { l: 60, r: 30, t: 20, b: 55 },
            hovermode: 'closest',
            paper_bgcolor: 'white',
            plot_bgcolor: '#f9fafb',
            legend: { orientation: 'v', x: 1.01, xanchor: 'left' },
            font: { family: '-apple-system, sans-serif', size: 12 }
        };
        const config = { responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['toImage'] };

        let traces = [];

        if (colorBy === '_none') {
            const main = points.filter(p => p.key !== currentKey);
            const sel  = points.filter(p => p.key === currentKey);
            traces.push({
                type: 'scatter', mode: 'markers', name: 'sources',
                x: main.map(p => p.x), y: main.map(p => p.y),
                customdata: main.map(p => p.key),
                hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<extra></extra>`,
                marker: { size: 7, color: '#3b82f6', opacity: 0.7 }
            });
            if (sel.length) traces.push({
                type: 'scatter', mode: 'markers', name: 'current',
                x: sel.map(p => p.x), y: sel.map(p => p.y),
                customdata: sel.map(p => p.key),
                hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<extra>current</extra>`,
                marker: { size: 13, color: '#ef4444', symbol: 'star', opacity: 1, line: { width: 1, color: '#7f1d1d' } }
            });

        } else if (colorBy === '_classification') {
            const groups = {};
            points.forEach(p => {
                if (!groups[p.cat]) groups[p.cat] = [];
                groups[p.cat].push(p);
            });
            const palette = ['#3b82f6','#10b981','#f59e0b','#ef4444','#8b5cf6','#ec4899','#06b6d4','#84cc16','#f97316','#6366f1'];
            Object.entries(groups).forEach(([cat, pts], gi) => {
                const main = pts.filter(p => p.key !== currentKey);
                const sel  = pts.filter(p => p.key === currentKey);
                const color = palette[gi % palette.length];
                if (main.length) traces.push({
                    type: 'scatter', mode: 'markers', name: cat,
                    x: main.map(p => p.x), y: main.map(p => p.y),
                    customdata: main.map(p => p.key),
                    hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<extra>${cat}</extra>`,
                    marker: { size: 7, color, opacity: 0.75 }
                });
                if (sel.length) traces.push({
                    type: 'scatter', mode: 'markers', name: `${cat} ★`,
                    x: sel.map(p => p.x), y: sel.map(p => p.y),
                    customdata: sel.map(p => p.key),
                    hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<extra>${cat} (current)</extra>`,
                    marker: { size: 13, color, symbol: 'star', opacity: 1, line: { width: 1, color: '#1f2937' } }
                });
            });

        } else {
            // Continuous color by a catalog column
            const main = points.filter(p => p.key !== currentKey);
            const sel  = points.filter(p => p.key === currentKey);
            const colorVals = main.map(p => p.row[colorBy] ?? null);
            traces.push({
                type: 'scatter', mode: 'markers', name: 'sources',
                x: main.map(p => p.x), y: main.map(p => p.y),
                customdata: main.map(p => p.key),
                hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<br>${colorBy}: %{marker.color:.4g}<extra></extra>`,
                marker: { size: 7, opacity: 0.8, color: colorVals, colorscale: 'Viridis',
                          showscale: true, colorbar: { title: colorBy, thickness: 14, len: 0.75 } }
            });
            if (sel.length) traces.push({
                type: 'scatter', mode: 'markers', name: 'current',
                x: sel.map(p => p.x), y: sel.map(p => p.y),
                customdata: sel.map(p => p.key),
                hovertemplate: `<b>%{customdata}</b><br>${xCol}: %{x:.4g}<br>${yCol}: %{y:.4g}<extra>current</extra>`,
                marker: { size: 13, color: '#ef4444', symbol: 'star', opacity: 1, line: { width: 1, color: '#7f1d1d' } }
            });
        }

        const plotDiv = document.getElementById('scatter-plot');
        Plotly.react(plotDiv, traces, layout, config);

        // Navigate to clicked point (only need to bind once; Plotly.react keeps handlers)
        plotDiv.removeAllListeners && plotDiv.removeAllListeners('plotly_click');
        plotDiv.on('plotly_click', function(data) {
            if (!data.points.length) return;
            const key = data.points[0].customdata;
            const idx = images.indexOf(key);
            if (idx === -1) return;
            closeScatterPlot();
            loadStateForImage(idx);
        });
    }

    // --- Shuffle ---

    function toggleShuffle() {
        shuffleMode = !shuffleMode;
        const btn = document.getElementById('btn-shuffle');
        btn.classList.toggle('active', shuffleMode);
        btn.textContent = shuffleMode ? 'Shuffle: On' : 'Shuffle';
        applyAllFilters(true);
    }

    // --- Autoscroll ---

    function renderAutoscrollBtn() {
        const btn = document.getElementById('btn-autoscroll');
        btn.classList.toggle('active', autoscrollMode);
        btn.textContent = autoscrollMode ? 'Autoscroll: On' : 'Autoscroll';
    }

    function toggleAutoscroll() {
        autoscrollMode = !autoscrollMode;
        renderAutoscrollBtn();
    }

    // --- Assignments ---

    async function fetchMyAssignment() {
        if (!currentFolder) return;
        try {
            const res = await fetch(`/api/assignments/me?folder=${encodeURIComponent(currentFolder)}`);
            const data = await res.json();
            currentAssignment = data.assignment;
            if (currentAssignment) {
                document.getElementById('assignment-notice').style.display = 'block';
                document.getElementById('assignment-count-display').textContent = currentAssignment.total.toLocaleString();
            }
        } catch(e) { console.error('Error fetching assignment:', e); }
    }

    function toggleAssignedOnly() {
        showAssignedOnly = document.getElementById('show-assigned-only').checked;
        applyAllFilters(true);
    }

    // --- Bulk Import ---

    function openImport() {
        if (!currentFolder) return;
        document.getElementById('import-modal').style.display = 'flex';
        resetImport();
    }

    function closeImport() {
        document.getElementById('import-modal').style.display = 'none';
    }

    document.getElementById('import-modal').addEventListener('click', e => {
        if (e.target === document.getElementById('import-modal')) closeImport();
    });

    function resetImport() {
        importParsedData = null;
        document.getElementById('import-file-input').value = '';
        document.getElementById('import-step-upload').style.display = 'block';
        document.getElementById('import-step-preview').style.display = 'none';
        document.getElementById('import-step-done').style.display = 'none';
        document.getElementById('import-parsing').style.display = 'none';
        document.getElementById('import-dropzone-inner').style.display = 'block';
        document.getElementById('import-parse-error').style.display = 'none';
    }

    async function handleImportFile() {
        const file = document.getElementById('import-file-input').files[0];
        if (!file) return;
        document.getElementById('import-dropzone-inner').style.display = 'none';
        document.getElementById('import-parsing').style.display = 'block';
        document.getElementById('import-parse-error').style.display = 'none';

        const formData = new FormData();
        formData.append('file', file);
        formData.append('folder', currentFolder);

        try {
            const res = await fetch('/api/bulk_import/preview', { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) {
                document.getElementById('import-parsing').style.display = 'none';
                document.getElementById('import-dropzone-inner').style.display = 'block';
                const errEl = document.getElementById('import-parse-error');
                errEl.textContent = 'Error: ' + data.error;
                errEl.style.display = 'block';
                return;
            }
            importParsedData = data;
            showImportPreview(data);
        } catch(e) {
            document.getElementById('import-parsing').style.display = 'none';
            document.getElementById('import-dropzone-inner').style.display = 'block';
            const errEl = document.getElementById('import-parse-error');
            errEl.textContent = 'Upload failed: ' + e.message;
            errEl.style.display = 'block';
        }
    }

    function showImportPreview(data) {
        const { matched, unmatched, total_file_rows } = data;
        const conflicts = matched.filter(r => r.conflict);

        document.getElementById('import-step-upload').style.display = 'none';
        document.getElementById('import-step-preview').style.display = 'block';

        const summary = document.getElementById('import-summary');
        summary.innerHTML = `
            <div class="import-stat"><span>Rows in file</span><b>${total_file_rows}</b></div>
            <div class="import-stat"><span>Matched to images in folder</span><b style="color:#059669;">${matched.length}</b></div>
            <div class="import-stat" style="${conflicts.length ? 'background:#fef3c7;' : ''}">
                <span>Already classified by me (conflicts)</span>
                <b style="color:${conflicts.length ? '#d97706' : '#6b7280'};">${conflicts.length}</b>
            </div>
            <div class="import-stat" style="${unmatched.length ? 'background:#fee2e2;' : ''}">
                <span>IDs not found in folder</span>
                <b style="color:${unmatched.length ? '#dc2626' : '#6b7280'};">${unmatched.length}</b>
            </div>`;

        const conflictSection = document.getElementById('import-conflict-section');
        if (conflicts.length > 0) {
            conflictSection.style.display = 'block';
            const previewEl = document.getElementById('import-preview-rows');
            previewEl.innerHTML = '<div style="font-weight:600; font-size:0.8rem; color:#374151; margin-bottom:6px;">Conflicts — existing &#8594; imported:</div>';
            conflicts.slice(0, 12).forEach(r => {
                const div = document.createElement('div');
                div.className = 'import-conflict-row';
                const fromColor = getCatColor(r.current_category);
                const toColor = getCatColor(r.category);
                div.innerHTML = `<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#6b7280;" title="${r.key}">${r.id}</span>
                    <span style="color:${fromColor};font-weight:600;">${r.current_category}</span>
                    <span style="color:#9ca3af;">&#8594;</span>
                    <span style="color:${toColor};font-weight:600;">${r.category}</span>`;
                previewEl.appendChild(div);
            });
            if (conflicts.length > 12) {
                const more = document.createElement('div');
                more.style.cssText = 'font-size:0.8rem; color:#6b7280; text-align:center; padding:4px 0;';
                more.textContent = `…and ${conflicts.length - 12} more`;
                previewEl.appendChild(more);
            }
        } else {
            conflictSection.style.display = 'none';
        }

        const btn = document.getElementById('btn-import-confirm');
        btn.disabled = matched.length === 0;
        btn.textContent = matched.length === 0 ? 'Nothing to import' : `Import ${matched.length} classification${matched.length !== 1 ? 's' : ''}`;
    }

    async function commitImport() {
        if (!importParsedData) return;
        const conflictMode = document.querySelector('input[name="conflict-mode"]:checked')?.value || 'skip';
        const rows = importParsedData.matched.map(r => ({ key: r.key, category: r.category, conflict: r.conflict }));

        const btn = document.getElementById('btn-import-confirm');
        btn.disabled = true;
        btn.textContent = 'Importing…';

        try {
            const res = await fetch('/api/bulk_import/commit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ conflict_mode: conflictMode, rows })
            });
            const result = await res.json();
            if (result.error) {
                alert('Import failed: ' + result.error);
                btn.disabled = false;
                btn.textContent = 'Import';
                return;
            }

            document.getElementById('import-step-preview').style.display = 'none';
            document.getElementById('import-step-done').style.display = 'block';
            document.getElementById('import-result').innerHTML = `
                <div class="import-stat" style="background:#d1fae5;"><span>Imported</span><b style="color:#059669;">${result.imported}</b></div>
                <div class="import-stat"><span>Skipped (conflicts)</span><b>${result.skipped}</b></div>
                ${result.errors.length ? `<div class="import-stat" style="background:#fee2e2;"><span>Errors</span><b style="color:#dc2626;">${result.errors.length}</b></div>` : ''}`;

            // Refresh image list to reflect new classifications
            if (result.imported > 0) await fetchImages();
        } catch(e) {
            alert('Import failed: ' + e.message);
            btn.disabled = false;
            btn.textContent = 'Import';
        }
    }

    async function loadStateForImage(index) {
        if (index < 0 || index >= images.length) return;
        currentIndex = index;
        const key = images[currentIndex];
        
        const displayname = key.split('/').pop(); 
        document.getElementById('filename-display').innerText = displayname;
        document.getElementById('counter-display').innerText = `${index + 1} / ${images.length}`;

        // Get Public URL
        const urlRes = await fetch(`/api/public_url?key=${encodeURIComponent(key)}`);
        const urlData = await urlRes.json();
        document.getElementById('main-image').src = urlData.url;

        // Get User Data
        const dataRes = await fetch(`/api/data?key=${encodeURIComponent(key)}`);
        const userData = await dataRes.json();

        document.getElementById('notes-input').value = userData.notes || '';
        highlightCategory(userData.category);

        // Tags on this image (not awaited; stale responses are discarded)
        fetchImageTags(key);

        // Show catalog properties for this image
        updateCatalogProps(key);

        // Update sky plot highlight if panel is open
        if (document.getElementById('sky-overlay').classList.contains('open')) {
            updateSkyHighlight();
        }

        // Refresh stats if open
        if (showStats) {
            fetchStats();
        }
    }

    function highlightCategory(cat) {
        document.querySelectorAll('.btn-cat').forEach(btn => {
            if (btn.dataset.category === cat) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
    }

    async function saveCurrentState() {
        if (currentIndex === -1) return;
        const key = images[currentIndex];
        const notes = document.getElementById('notes-input').value;
        const activeBtn = document.querySelector('.btn-cat.active');
        const category = activeBtn ? activeBtn.dataset.category : '';

        await fetch('/api/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ key, notes, category })
        });
        
        const t = document.getElementById('toast');
        t.style.opacity = '1';
        setTimeout(() => t.style.opacity = '0', 1000);
        
        // Refresh stats if open (so your own vote counts immediately)
        if (showStats) {
            fetchStats();
        }
    }

    async function selectCategory(cat) {
        highlightCategory(cat);
        await saveCurrentState();
        // Autoscroll: once classified, advance to the next image (stop at the end)
        if (autoscrollMode && currentIndex >= 0 && currentIndex < images.length - 1) {
            loadStateForImage(currentIndex + 1);
        }
    }

    async function navigate(dir) {
        if (currentIndex === -1) return;
        await saveCurrentState();
        let newIndex = currentIndex + dir;
        if (newIndex >= 0 && newIndex < images.length) {
            loadStateForImage(newIndex);
        }
    }

    function exportClassifications(fmt) {
        if (!currentFolder) return;
        const allUsers = document.getElementById('export-all-users').checked;
        const url = `/api/export?folder=${encodeURIComponent(currentFolder)}&format=${fmt}&all_users=${allUsers}`;
        window.location.href = url;
    }

    init();
</script>
</body>
</html>
"""

# --- Routes ---

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin:
            abort(403)
        return f(*args, **kwargs)
    return decorated


@app.route('/admin')
@login_required
@admin_required
def admin():
    return render_template_string(ADMIN_TEMPLATE)


@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE, is_admin=current_user.is_admin)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        passphrase = request.form.get('passphrase')
        
        user = User.query.filter_by(username=username).first()
        
        if user:
            # Login Existing User
            if check_password_hash(user.password_hash, password):
                login_user(user)
                return redirect(url_for('index'))
            else:
                flash("Invalid password.")
        else:
            # Register New User
            if REGISTRATION_PASSPHRASE and passphrase != REGISTRATION_PASSPHRASE:
                flash("Invalid Registration Key. Please contact administrator.")
            else:
                # Create Account — first registered user becomes admin,
                # as does anyone listed in ADMIN_USERNAMES
                is_first = User.query.count() == 0
                new_user = User(username=username, password_hash=generate_password_hash(password),
                                is_admin=is_first or username in ADMIN_USERNAMES)
                db.session.add(new_user)
                db.session.commit()
                login_user(new_user)
                flash("Account created successfully!")
                return redirect(url_for('index'))
            
    return render_template_string(LOGIN_TEMPLATE, require_passphrase=bool(REGISTRATION_PASSPHRASE))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    is_sqlite = app.config['SQLALCHEMY_DATABASE_URI'].startswith('sqlite')
    return render_template_string(APP_TEMPLATE, is_sqlite=is_sqlite,
                                   is_admin=current_user.is_admin)

@app.route('/api/users')
@login_required
def get_users():
    users = User.query.filter(User.id != current_user.id).order_by(User.username).all()
    return jsonify([{'id': u.id, 'username': u.username} for u in users])

@app.route('/api/config')
@login_required
def get_config():
    base_config = {"categories": ["Good", "Bad"]}
    if os.path.exists("config.yaml"):
        with open("config.yaml", 'r') as f:
            base_config = yaml.safe_load(f)

    folder = request.args.get('folder', '').strip()
    if folder:
        candidates = [
            f"classifier/{folder}/config.yaml",
            f"{folder}/config.yaml",
        ]
        for key in candidates:
            try:
                obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
                folder_config = yaml.safe_load(obj['Body'].read())
                if folder_config and 'categories' in folder_config:
                    logger.info(f"Using folder-specific config from R2 key: {key}")
                    return jsonify(folder_config)
            except ClientError as e:
                if e.response['Error']['Code'] not in ('NoSuchKey', '404'):
                    logger.error(f"S3 error fetching config {key}: {e}")
            except Exception as e:
                logger.error(f"Error fetching folder config {key}: {e}")

    return jsonify(base_config)

def list_r2_folders() -> List[str]:
    """Lists subdirectories under classifier/ (falling back to the bucket root)."""
    folders = []

    # 1. Try finding folders under 'classifier/'
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_NAME, 
            Prefix='classifier/', 
            Delimiter='/'
        )
        for p in response.get('CommonPrefixes', []):
            raw_prefix = p['Prefix'] 
            folder_name = raw_prefix.rstrip('/').split('/')[-1]
            if folder_name:
                folders.append(folder_name)
    except Exception as e:
        logger.error(f"Error listing 'classifier/' prefix: {e}", exc_info=True)

    # 2. Fallback: If no folders found, look at root just in case user structure is different
    if not folders:
        logger.info("No folders found in classifier/, trying root...")
        try:
            response = s3_client.list_objects_v2(
                Bucket=BUCKET_NAME, 
                Delimiter='/'
            )
            for p in response.get('CommonPrefixes', []):
                raw_prefix = p['Prefix']
                folder_name = raw_prefix.rstrip('/')
                if folder_name and folder_name != 'classifier': # avoid duplicate if listing root
                    folders.append(folder_name)
        except Exception as e:
             logger.error(f"Error listing root prefix: {e}", exc_info=True)

    folders.sort()
    return folders


@app.route('/api/folders')
@login_required
def get_folders():
    """Lists folders, excluding hidden ones for non-admin users."""
    logger.info("Request: /api/folders started")
    folders = list_r2_folders()
    if not current_user.is_admin:
        hidden = {h.folder for h in HiddenFolder.query.all()}
        folders = [f for f in folders if f not in hidden]
    logger.info(f"Returning folders: {folders}")
    return jsonify(folders)

@app.route('/api/catalog')
@login_required
def get_catalog():
    """Return catalog.fits data for a folder as JSON, or {available: false} if not found."""
    folder = request.args.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    catalog = load_catalog_for_folder(folder)
    if catalog is None:
        return jsonify({'available': False})

    return jsonify({'available': True, 'columns': catalog['columns'], 'rows': catalog['rows']})


@app.route('/api/catalog/mask', methods=['POST'])
@login_required
def catalog_mask():
    """Evaluate a mask expression against the catalog and return the passing IDs."""
    data = request.get_json(force=True) or {}
    folder = data.get('folder', '').strip()
    expr = data.get('expr', '').strip()
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    # Ensure catalog is loaded so the table cache is populated
    load_catalog_for_folder(folder)
    table = _catalog_table_cache.get(folder)
    if table is None:
        return jsonify({'error': 'No catalog available for this folder'}), 404

    try:
        parser = TableQueryParser(table)
        mask = parser.parse(expr)
        ids = [str(_serialize_fits_value(table['ID'][i])) for i in range(len(table)) if mask[i]]
        return jsonify({'count': int(mask.sum()), 'total': len(table), 'ids': ids})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


FULL_SKY_DEG2 = 41252.96124941927  # 4*pi steradians expressed in square degrees


@app.route('/api/catalog/vmax', methods=['POST'])
@login_required
def catalog_vmax():
    """Compute a 1/Vmax luminosity function over the catalog.

    Shared-shell Vmax: every selected galaxy is assumed detectable out to
    z_max, so Vmax is the comoving volume of the [z_min, z_max] shell over the
    survey area, identical for all galaxies.  phi(M) = N(M) / (Vmax * dM),
    with Poisson errors sqrt(N) / (Vmax * dM).

    Accepts up to two mask expressions (``exprs``) so the client can overlay
    two cuts.  An empty expression means "no cut".
    """
    data = request.get_json(force=True) or {}
    folder = (data.get('folder') or '').strip()
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    load_catalog_for_folder(folder)
    table = _catalog_table_cache.get(folder)
    if table is None:
        return jsonify({'error': 'No catalog available for this folder'}), 404

    mag_col = (data.get('mag_col') or '').strip()
    z_col = (data.get('z_col') or '').strip()
    if mag_col not in table.colnames:
        return jsonify({'error': f"Magnitude column '{mag_col}' not found"}), 400
    if z_col not in table.colnames:
        return jsonify({'error': f"Redshift column '{z_col}' not found"}), 400

    try:
        z_min = float(data.get('z_min'))
        z_max = float(data.get('z_max'))
        area_deg2 = float(data.get('area_deg2'))
        bin_width = float(data.get('bin_width'))
        H0 = float(data.get('H0', 70.0))
        Om0 = float(data.get('Om0', 0.3))
    except (TypeError, ValueError):
        return jsonify({'error': 'z range, area, bin width and cosmology must be numbers'}), 400

    if not (z_min >= 0 and z_max > z_min):
        return jsonify({'error': 'Require 0 <= z_min < z_max'}), 400
    if area_deg2 <= 0:
        return jsonify({'error': 'Survey area must be positive'}), 400
    if bin_width <= 0:
        return jsonify({'error': 'Bin width must be positive'}), 400

    # --- Comoving shell volume over the survey area ---
    try:
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        vc_max = cosmo.comoving_volume(z_max).to(u.Mpc ** 3).value
        vc_min = cosmo.comoving_volume(z_min).to(u.Mpc ** 3).value
    except Exception as exc:
        return jsonify({'error': f'Cosmology error: {exc}'}), 400

    sky_frac = area_deg2 / FULL_SKY_DEG2
    vmax = (vc_max - vc_min) * sky_frac
    if vmax <= 0:
        return jsonify({'error': 'Computed Vmax is non-positive'}), 400

    mag_all = np.asarray(table[mag_col], dtype=float)
    z_all = np.asarray(table[z_col], dtype=float)
    ids_all = [str(_serialize_fits_value(table['ID'][i])) for i in range(len(table))]
    z_select = np.isfinite(z_all) & (z_all >= z_min) & (z_all <= z_max)

    # Optional parent-sample restriction to a set of catalog IDs (current sidebar
    # filter, or a classification such as "robust").  None = no restriction.
    restrict = data.get('restrict_ids', None)
    if restrict is not None:
        rset = {str(x) for x in restrict}
        z_select = z_select & np.array([rid in rset for rid in ids_all], dtype=bool)

    exprs = data.get('exprs') or ['']
    if not isinstance(exprs, list):
        exprs = [exprs]
    exprs = exprs[:2] if exprs else ['']

    parser = TableQueryParser(table)
    series = []
    galaxies = []
    for expr in exprs:
        expr = (expr or '').strip()
        try:
            emask = np.asarray(parser.parse(expr), dtype=bool)
        except Exception as exc:
            return jsonify({'error': f"Expression '{expr}': {exc}"}), 400

        sel = z_select & np.isfinite(mag_all) & emask
        n = int(sel.sum())
        if n == 0:
            series.append({'expr': expr, 'count': 0, 'bins': []})
            galaxies.append([])
            continue

        mags = mag_all[sel]
        idx = np.where(sel)[0]
        lo = math.floor(mags.min() / bin_width) * bin_width
        hi = math.ceil(mags.max() / bin_width) * bin_width
        nbins = max(1, int(round((hi - lo) / bin_width)))
        edges = lo + bin_width * np.arange(nbins + 1)
        counts, _ = np.histogram(mags, bins=edges)

        norm = vmax * bin_width
        bins = []
        for b in range(nbins):
            c = int(counts[b])
            bins.append({
                'center': round(float(edges[b] + bin_width / 2.0), 4),
                'lo': round(float(edges[b]), 4),
                'hi': round(float(edges[b + 1]), 4),
                'count': c,
                'phi': c / norm,
                'phi_err': math.sqrt(c) / norm,
            })
        series.append({'expr': expr, 'count': n, 'bins': bins})

        bin_of = np.clip(np.floor((mags - lo) / bin_width).astype(int), 0, nbins - 1)
        gl = [{
            'id': ids_all[gi],
            'mag': round(float(mag_all[gi]), 4),
            'z': round(float(z_all[gi]), 4),
            'bin': int(bin_of[k]),
        } for k, gi in enumerate(idx)]
        galaxies.append(gl)

    return jsonify({
        'vmax': vmax,
        'volume_mpc3': vc_max - vc_min,
        'sky_frac': sky_frac,
        'series': series,
        'galaxies': galaxies,
    })


def folder_hidden_for_current_user(folder: str) -> bool:
    """True if the folder is hidden and the current user is not an admin."""
    if current_user.is_admin:
        return False
    return HiddenFolder.query.filter_by(folder=folder).first() is not None


def folder_key_condition(column, folder: str):
    """SQL condition matching image keys that live in *folder* (either bucket layout)."""
    return db.or_(column.like(f"classifier/{folder}/%"), column.like(f"{folder}/%"))


def list_all_image_keys(prefix: str) -> List[str]:
    """Paginate through all R2 objects under *prefix* and return image keys."""
    keys: List[str] = []
    kwargs: Dict = {'Bucket': BUCKET_NAME, 'Prefix': prefix}
    while True:
        resp = s3_client.list_objects_v2(**kwargs)
        for obj in resp.get('Contents', []):
            if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg', '.fits')):
                keys.append(obj['Key'])
        if resp.get('IsTruncated'):
            kwargs['ContinuationToken'] = resp['NextContinuationToken']
        else:
            break
    return keys


@app.route('/api/images')
@login_required
def get_images():
    folder = request.args.get('folder')
    logger.info(f"Request: /api/images started for folder: {folder}")
    if not folder:
        return jsonify([])
    if folder_hidden_for_current_user(folder):
        return jsonify([])

    try:
        prefix = f"classifier/{folder}/"
        all_keys = [k for k in list_all_image_keys(prefix)
                    if k.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Fallback: folder at bucket root
        if not all_keys:
            logger.info(f"No contents in {prefix}, trying root prefix {folder}/")
            prefix = f"{folder}/"
            all_keys = [k for k in list_all_image_keys(prefix)
                        if k.lower().endswith(('.png', '.jpg', '.jpeg'))]

        all_keys.sort()
        logger.info(f"Found {len(all_keys)} images")
    except Exception as e:
        logger.error(f"Error listing images: {e}", exc_info=True)
        return jsonify([])

    # Apply Classification Filters
    filter_val = request.args.get('filter', 'All')

    if filter_val == 'All':
        return jsonify(all_keys)

    # Multi-user filters (don't depend on a single user's classifications)
    if filter_val == 'NoVotes':
        try:
            classified_any = Classification.query.filter(
                Classification.category != None,
                Classification.category != '',
                db.or_(
                    Classification.image_key.like(f"classifier/{folder}/%"),
                    Classification.image_key.like(f"{folder}/%"),
                )
            ).with_entities(Classification.image_key).distinct().all()
            classified_keys = {r.image_key for r in classified_any}
            filtered = [key for key in all_keys if key not in classified_keys]
        except Exception as e:
            logger.error(f"Error computing no-votes filter: {e}", exc_info=True)
            filtered = []
        return jsonify(filtered)

    # Tag filters (tags are shared across users)
    if filter_val == 'Untagged' or filter_val.startswith('tag:'):
        try:
            q = ImageTag.query.filter(folder_key_condition(ImageTag.image_key, folder))
            if filter_val.startswith('tag:'):
                q = q.filter(ImageTag.tag_id == int(filter_val[4:]))
            tagged_keys = {r.image_key for r in q.with_entities(ImageTag.image_key).distinct().all()}
        except Exception as e:
            logger.error(f"Error computing tag filter: {e}", exc_info=True)
            tagged_keys = set()
        if filter_val == 'Untagged':
            return jsonify([key for key in all_keys if key not in tagged_keys])
        return jsonify([key for key in all_keys if key in tagged_keys])

    if filter_val in ('Agree', 'Disagree'):
        try:
            records = Classification.query.filter(
                Classification.category != None,
                Classification.category != '',
                db.or_(
                    Classification.image_key.like(f"classifier/{folder}/%"),
                    Classification.image_key.like(f"{folder}/%"),
                )
            ).all()
            key_user_cat = {}
            for r in records:
                if r.image_key not in key_user_cat:
                    key_user_cat[r.image_key] = {}
                key_user_cat[r.image_key][r.user_id] = r.category
            target_keys = set()
            for img_key, user_cats in key_user_cat.items():
                if len(user_cats) < 2:
                    continue
                unique_cats = set(user_cats.values())
                if filter_val == 'Agree' and len(unique_cats) == 1:
                    target_keys.add(img_key)
                elif filter_val == 'Disagree' and len(unique_cats) > 1:
                    target_keys.add(img_key)
        except Exception as e:
            logger.error(f"Error computing agreement filter: {e}", exc_info=True)
            target_keys = set()
        return jsonify([key for key in all_keys if key in target_keys])

    # Resolve which user's classifications to filter by
    filter_user_id = current_user.id
    filter_user_id_param = request.args.get('filter_user_id', '')
    if filter_user_id_param:
        try:
            filter_user_id = int(filter_user_id_param)
        except (ValueError, TypeError):
            pass

    try:
        user_classes = Classification.query.filter_by(user_id=filter_user_id).all()
        class_map = {c.image_key: c.category for c in user_classes}
    except Exception as e:
        logger.error(f"Error fetching user classifications: {e}", exc_info=True)
        class_map = {}

    filtered = []
    for key in all_keys:
        cat = class_map.get(key, "")
        if filter_val == 'Uncategorized':
            if not cat: filtered.append(key)
        elif filter_val == 'Classified':
            if cat: filtered.append(key)
        elif cat == filter_val:
            filtered.append(key)

    return jsonify(filtered)

@app.route('/api/public_url')
@login_required
def get_public_url():
    key = request.args.get('key')
    if not key: return jsonify({'error': 'No key'}), 400
    
    # Construct the public URL
    base = R2_PUBLIC_DOMAIN.rstrip('/')
    if not base.startswith('http'):
        base = f"https://{base}"
        
    url = f"{base}/{key}"
    return jsonify({'url': url})

@app.route('/api/classifications')
@login_required
def get_classifications():
    """Return all of the current user's classifications for a given folder as {image_key: category}."""
    folder = request.args.get('folder')
    if not folder:
        return jsonify({})
    try:
        # Match both possible prefixes (classifier/{folder}/ and {folder}/)
        records = Classification.query.filter(
            Classification.user_id == current_user.id,
            Classification.category != None,
            Classification.category != '',
            db.or_(
                Classification.image_key.like(f"classifier/{folder}/%"),
                Classification.image_key.like(f"{folder}/%"),
            )
        ).all()
        return jsonify({r.image_key: r.category for r in records})
    except Exception as e:
        logger.error(f"Error fetching classifications for folder {folder}: {e}", exc_info=True)
        return jsonify({})


# --- Tags (shared, user-defined image groupings) ---

TAG_PALETTE = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899',
               '#06b6d4', '#84cc16', '#f97316', '#6366f1', '#14b8a6', '#a855f7']


def tag_counts(folder: Optional[str]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Return ({tag_id: images tagged in folder}, {tag_id: images tagged anywhere})."""
    total_rows = db.session.query(ImageTag.tag_id, db.func.count(ImageTag.id)) \
        .group_by(ImageTag.tag_id).all()
    total = {tid: n for tid, n in total_rows}
    if not folder:
        return dict(total), total
    folder_rows = db.session.query(ImageTag.tag_id, db.func.count(ImageTag.id)) \
        .filter(folder_key_condition(ImageTag.image_key, folder)) \
        .group_by(ImageTag.tag_id).all()
    return {tid: n for tid, n in folder_rows}, total


def serialize_tag(tag: 'Tag', folder_counts: Dict[int, int], total_counts: Dict[int, int]) -> Dict:
    return {
        'id': tag.id,
        'name': tag.name,
        'color': tag.color,
        'created_by': tag.created_by,
        'count': folder_counts.get(tag.id, 0),
        'total': total_counts.get(tag.id, 0),
        'can_delete': bool(current_user.is_admin or tag.created_by == current_user.id),
    }


@app.route('/api/tags')
@login_required
def list_tags():
    """All defined tags, with usage counts for the given folder and overall."""
    folder = request.args.get('folder') or None
    try:
        tags = Tag.query.order_by(db.func.lower(Tag.name)).all()
        folder_counts, total_counts = tag_counts(folder)
        return jsonify({'tags': [serialize_tag(t, folder_counts, total_counts) for t in tags]})
    except Exception as e:
        logger.error(f"Error listing tags: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/tags', methods=['POST'])
@login_required
def create_tag():
    """Create a tag (any user). Names are unique case-insensitively; an existing match is returned as-is."""
    data = request.json or {}
    name = ' '.join(str(data.get('name', '')).split())  # trim + collapse whitespace
    folder = data.get('folder') or None
    if not name:
        return jsonify({'error': 'Tag name is required'}), 400
    if len(name) > 100:
        return jsonify({'error': 'Tag name must be 100 characters or fewer'}), 400
    try:
        tag = Tag.query.filter(db.func.lower(Tag.name) == name.lower()).first()
        created = False
        if not tag:
            # Least-used palette colour keeps tags visually distinct
            used = [t.color for t in Tag.query.all()]
            color = min(TAG_PALETTE, key=lambda c: (used.count(c), TAG_PALETTE.index(c)))
            tag = Tag(name=name, color=color, created_by=current_user.id)
            db.session.add(tag)
            db.session.commit()
            created = True
        folder_counts, total_counts = tag_counts(folder)
        return jsonify({'tag': serialize_tag(tag, folder_counts, total_counts), 'created': created}), \
            (201 if created else 200)
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error creating tag '{name}': {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/tags/<int:tag_id>', methods=['DELETE'])
@login_required
def delete_tag(tag_id):
    """Delete a tag and all of its image memberships. Allowed for the creator or an admin."""
    tag = Tag.query.get(tag_id)
    if not tag:
        return jsonify({'error': 'Tag not found'}), 404
    if not (current_user.is_admin or tag.created_by == current_user.id):
        return jsonify({'error': 'Only the tag creator or an admin can delete this tag'}), 403
    try:
        # Explicit delete: SQLite ignores ON DELETE CASCADE unless foreign_keys is enabled
        removed = ImageTag.query.filter_by(tag_id=tag_id).delete(synchronize_session=False)
        db.session.delete(tag)
        db.session.commit()
        return jsonify({'status': 'success', 'removed': removed})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting tag {tag_id}: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/image_tags')
@login_required
def get_image_tags():
    """Tags applied to a single image, with who applied each."""
    key = request.args.get('key')
    if not key:
        return jsonify({'error': 'No key provided'}), 400
    try:
        rows = db.session.query(ImageTag, Tag, User.username) \
            .join(Tag, ImageTag.tag_id == Tag.id) \
            .outerjoin(User, ImageTag.user_id == User.id) \
            .filter(ImageTag.image_key == key) \
            .order_by(db.func.lower(Tag.name)).all()
        return jsonify({'tags': [
            {'id': t.id, 'name': t.name, 'color': t.color, 'username': username,
             'created_at': it.created_at.isoformat() if it.created_at else None}
            for it, t, username in rows
        ]})
    except Exception as e:
        logger.error(f"Error fetching tags for key {key}: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


def _parse_image_tag_request():
    """Shared parsing for add/remove. Returns (tag, keys, error) where error is (payload, status) or None."""
    data = request.json or {}
    keys = data.get('keys')
    if keys is None and data.get('key'):
        keys = [data['key']]
    keys = list(dict.fromkeys(k for k in (keys or []) if isinstance(k, str) and k))
    try:
        tag_id = int(data.get('tag_id'))
    except (TypeError, ValueError):
        return None, [], ({'error': 'tag_id is required'}, 400)
    tag = Tag.query.get(tag_id)
    if not tag:
        return None, [], ({'error': 'Tag not found'}, 404)
    if not keys:
        return tag, [], ({'error': 'No image keys provided'}, 400)
    return tag, keys, None


@app.route('/api/image_tags/add', methods=['POST'])
@login_required
def add_image_tags():
    """Apply a tag to one or more images. Any user may add; images already carrying the tag are skipped."""
    tag, keys, err = _parse_image_tag_request()
    if err:
        return jsonify(err[0]), err[1]
    try:
        existing = {r.image_key for r in ImageTag.query.filter_by(tag_id=tag.id)
                    .with_entities(ImageTag.image_key).all()}
        new_keys = [k for k in keys if k not in existing]
        for k in new_keys:
            db.session.add(ImageTag(tag_id=tag.id, image_key=k, user_id=current_user.id))
        db.session.commit()
        return jsonify({'status': 'success', 'changed': len(new_keys), 'username': current_user.username})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding tag {tag.id} to {len(keys)} image(s): {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/image_tags/remove', methods=['POST'])
@login_required
def remove_image_tags():
    """Remove a tag from one or more images. Any user may remove: tags are shared, ad hoc groupings."""
    tag, keys, err = _parse_image_tag_request()
    if err:
        return jsonify(err[0]), err[1]
    try:
        removed = 0
        # Chunk the IN clause to stay under SQLite's bound-parameter limit
        for i in range(0, len(keys), 500):
            removed += ImageTag.query.filter(
                ImageTag.tag_id == tag.id, ImageTag.image_key.in_(keys[i:i + 500])
            ).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({'status': 'success', 'changed': removed})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error removing tag {tag.id} from {len(keys)} image(s): {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


@app.route('/api/tags/<int:tag_id>/export')
@login_required
def export_tag(tag_id):
    """CSV of every image in a folder carrying this tag, joined with catalog columns when available."""
    import csv as csv_mod
    folder = request.args.get('folder')
    tag = Tag.query.get(tag_id)
    if not tag:
        return jsonify({'error': 'Tag not found'}), 404
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400
    if folder_hidden_for_current_user(folder):
        abort(403)
    try:
        members = db.session.query(ImageTag, User.username) \
            .outerjoin(User, ImageTag.user_id == User.id) \
            .filter(ImageTag.tag_id == tag_id, folder_key_condition(ImageTag.image_key, folder)) \
            .order_by(ImageTag.image_key).all()
    except Exception as e:
        logger.error(f"Tag export DB error: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500

    catalog = load_catalog_for_folder(folder)
    cat_cols = [c for c in catalog['columns'] if c != 'ID'] if catalog else []
    base_cols = ['ID', 'image_key', 'tag', 'tagged_by', 'tagged_at']
    rows = []
    for it, username in members:
        stem = os.path.splitext(it.image_key.split('/')[-1])[0]
        row: Dict = {'ID': stem, 'image_key': it.image_key, 'tag': tag.name,
                     'tagged_by': username or '',
                     'tagged_at': it.created_at.isoformat() if it.created_at else ''}
        if catalog:
            cat_row = catalog['rows'].get(stem, {})
            for col in cat_cols:
                val = cat_row.get(col)
                row[col] = '' if val is None else val
        rows.append(row)

    buf = io.StringIO()
    writer = csv_mod.DictWriter(buf, fieldnames=base_cols + cat_cols)
    writer.writeheader()
    writer.writerows(rows)
    safe_folder = folder.replace('/', '_')
    safe_tag = ''.join(ch if ch.isalnum() or ch in '-_' else '_' for ch in tag.name)
    return app.response_class(
        buf.getvalue(), mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{safe_folder}_tag_{safe_tag}.csv"'}
    )


@app.route('/api/export')
@login_required
def export_classifications():
    """Download classifications for a folder as CSV or FITS, optionally joined with catalog columns."""
    import csv as csv_mod
    folder = request.args.get('folder')
    fmt = request.args.get('format', 'csv').lower()
    all_users = request.args.get('all_users', 'false').lower() == 'true'

    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    try:
        q = Classification.query.filter(
            db.or_(
                Classification.image_key.like(f"classifier/{folder}/%"),
                Classification.image_key.like(f"{folder}/%"),
            )
        )
        if not all_users:
            q = q.filter_by(user_id=current_user.id)
        records = q.order_by(Classification.image_key).all()
    except Exception as e:
        logger.error(f"Export DB error: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500

    # Load catalog once so we can join columns
    catalog = load_catalog_for_folder(folder)
    cat_cols = [c for c in catalog['columns'] if c != 'ID'] if catalog else []

    # Username lookup cache
    user_cache: Dict[int, str] = {}

    def username_of(uid):
        if uid not in user_cache:
            u = User.query.get(uid)
            user_cache[uid] = u.username if u else str(uid)
        return user_cache[uid]

    # Tag names per image (shared across users), joined with ';' in the export
    tag_map: Dict[str, List[str]] = {}
    try:
        tag_rows = db.session.query(ImageTag.image_key, Tag.name) \
            .join(Tag, ImageTag.tag_id == Tag.id) \
            .filter(folder_key_condition(ImageTag.image_key, folder)) \
            .order_by(db.func.lower(Tag.name)).all()
        for image_key, tag_name in tag_rows:
            tag_map.setdefault(image_key, []).append(tag_name)
    except Exception as e:
        logger.error(f"Export tag lookup error: {e}", exc_info=True)

    # Build row dicts
    rows = []
    for r in records:
        stem = os.path.splitext(r.image_key.split('/')[-1])[0]
        row: Dict = {}
        if all_users:
            row['username'] = username_of(r.user_id)
        row['ID'] = stem
        row['image_key'] = r.image_key
        row['category'] = r.category or ''
        row['notes'] = (r.notes or '').replace('\n', ' ')
        row['tags'] = ';'.join(tag_map.get(r.image_key, []))
        row['timestamp'] = r.timestamp.isoformat() if r.timestamp else ''
        if catalog:
            cat_row = catalog['rows'].get(stem, {})
            for col in cat_cols:
                val = cat_row.get(col)
                row[col] = '' if val is None else val
        rows.append(row)

    safe_folder = folder.replace('/', '_')

    if fmt == 'fits':
        try:
            import numpy as np
            from astropy.table import Table
            from astropy.io.fits import Column as FitsColumn

            if not rows:
                buf = io.BytesIO()
                Table().write(buf, format='fits', overwrite=True)
                buf.seek(0)
            else:
                col_names = list(rows[0].keys())
                col_data: Dict = {c: [] for c in col_names}
                for row in rows:
                    for c in col_names:
                        col_data[c].append(row[c])

                tbl_dict = {}
                for c in col_names:
                    vals = col_data[c]
                    # Detect numeric columns (all values are int or float, or empty string)
                    numeric = all(isinstance(v, (int, float)) or v == '' for v in vals)
                    if numeric and any(isinstance(v, (int, float)) for v in vals):
                        arr = np.array([float(v) if v != '' else np.nan for v in vals], dtype=np.float64)
                        tbl_dict[c] = arr
                    else:
                        tbl_dict[c] = [str(v) for v in vals]

                buf = io.BytesIO()
                Table(tbl_dict).write(buf, format='fits', overwrite=True)
                buf.seek(0)

            return app.response_class(
                buf.read(),
                mimetype='application/octet-stream',
                headers={'Content-Disposition': f'attachment; filename="{safe_folder}_classifications.fits"'}
            )
        except Exception as e:
            logger.error(f"FITS export error: {e}", exc_info=True)
            return jsonify({'error': f'FITS export failed: {e}'}), 500

    else:  # CSV
        buf = io.StringIO()
        if rows:
            writer = csv_mod.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return app.response_class(
            buf.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{safe_folder}_classifications.csv"'}
        )


@app.route('/api/data')
@login_required
def get_data():
    key = request.args.get('key')
    try:
        record = Classification.query.filter_by(
            user_id=current_user.id, 
            image_key=key
        ).first()
        
        if record:
            return jsonify({'category': record.category, 'notes': record.notes})
        return jsonify({'category': '', 'notes': ''})
    except Exception as e:
        logger.error(f"Error fetching data for key {key}: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500

@app.route('/api/stats')
@login_required
def get_stats():
    key = request.args.get('key')
    if not key: return jsonify({'error': 'No key provided'}), 400

    try:
        count_results = db.session.query(Classification.category, db.func.count(Classification.category)) \
            .filter_by(image_key=key) \
            .group_by(Classification.category).all()
        counts = {r[0]: r[1] for r in count_results if r[0]}

        vote_records = db.session.query(Classification, User).join(
            User, Classification.user_id == User.id
        ).filter(
            Classification.image_key == key,
            Classification.category != None,
            Classification.category != ''
        ).order_by(User.username).all()
        votes = [{'username': u.username, 'category': c.category} for c, u in vote_records]

        return jsonify({'counts': counts, 'votes': votes})
    except Exception as e:
        logger.error(f"Error fetching stats for key {key}: {e}", exc_info=True)
        return jsonify({'error': 'Database stats error'}), 500

@app.route('/api/save', methods=['POST'])
@login_required
def save_data():
    try:
        data = request.json
        key = data.get('key')
        
        record = Classification.query.filter_by(
            user_id=current_user.id, 
            image_key=key
        ).first()

        if not record:
            record = Classification(user_id=current_user.id, image_key=key)
            db.session.add(record)
        
        record.category = data.get('category')
        record.notes = data.get('notes')
        db.session.commit()
        
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Error saving data: {e}", exc_info=True)
        return jsonify({'error': 'Database save error'}), 500

@app.route('/api/db_backup')
@login_required
def db_backup():
    """Download a full backup of the classification database."""
    fmt = request.args.get('format', 'json')
    ts = dt.utcnow().strftime('%Y%m%d_%H%M%S')

    if fmt == 'sqlite':
        db_url = app.config['SQLALCHEMY_DATABASE_URI']
        if not db_url.startswith('sqlite'):
            return jsonify({'error': 'SQLite backup only available when using a SQLite database'}), 400
        db_path = db_url[len('sqlite:///'):]
        if not os.path.isabs(db_path):
            db_path = os.path.join(os.getcwd(), db_path)
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(suffix='.db')
            os.close(tmp_fd)
            src = sqlite3.connect(db_path)
            dst = sqlite3.connect(tmp_path)
            src.backup(dst)
            dst.close()
            src.close()
            with open(tmp_path, 'rb') as f:
                data = f.read()
            os.unlink(tmp_path)
            return app.response_class(
                data, mimetype='application/octet-stream',
                headers={'Content-Disposition': f'attachment; filename="classifier_db_{ts}.db"'}
            )
        except Exception as e:
            logger.error(f"SQLite backup error: {e}", exc_info=True)
            return jsonify({'error': f'SQLite backup failed: {e}'}), 500

    # JSON backup — works for any database backend
    try:
        users = User.query.order_by(User.id).all()
        classifications = Classification.query.order_by(Classification.id).all()
        payload = {
            'exported_at': dt.utcnow().isoformat() + 'Z',
            'tables': {
                'users': [
                    {'id': u.id, 'username': u.username, 'password_hash': u.password_hash}
                    for u in users
                ],
                'classifications': [
                    {
                        'id': c.id,
                        'user_id': c.user_id,
                        'image_key': c.image_key,
                        'category': c.category,
                        'notes': c.notes,
                        'timestamp': c.timestamp.isoformat() if c.timestamp else None,
                    }
                    for c in classifications
                ],
                'tags': [
                    {'id': t.id, 'name': t.name, 'color': t.color, 'created_by': t.created_by,
                     'created_at': t.created_at.isoformat() if t.created_at else None}
                    for t in Tag.query.order_by(Tag.id).all()
                ],
                'image_tags': [
                    {'id': it.id, 'tag_id': it.tag_id, 'image_key': it.image_key, 'user_id': it.user_id,
                     'created_at': it.created_at.isoformat() if it.created_at else None}
                    for it in ImageTag.query.order_by(ImageTag.id).all()
                ],
            }
        }
        buf = json.dumps(payload, indent=2, ensure_ascii=False)
        return app.response_class(
            buf, mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="classifier_db_{ts}.json"'}
        )
    except Exception as e:
        logger.error(f"JSON backup error: {e}", exc_info=True)
        return jsonify({'error': f'Backup failed: {e}'}), 500


@app.route('/api/admin/users')
@login_required
@admin_required
def admin_get_users():
    users = User.query.order_by(User.username).all()
    result = []
    for u in users:
        count = Classification.query.filter(
            Classification.user_id == u.id,
            Classification.category != None,
            Classification.category != ''
        ).count()
        result.append({'id': u.id, 'username': u.username, 'is_admin': u.is_admin,
                       'classification_count': count})
    return jsonify(result)


@app.route('/api/admin/users/<int:uid>/toggle_admin', methods=['POST'])
@login_required
@admin_required
def admin_toggle_admin(uid):
    if uid == current_user.id:
        return jsonify({'error': 'Cannot change your own admin status'}), 400
    user = User.query.get(uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.is_admin = not user.is_admin
    db.session.commit()
    return jsonify({'is_admin': user.is_admin})


@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_user(uid):
    if uid == current_user.id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    user = User.query.get(uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    deleted_name = user.username
    try:
        Classification.query.filter_by(user_id=uid).delete()
        Assignment.query.filter_by(user_id=uid).delete()
        Assignment.query.filter_by(created_by=uid).update({'created_by': None})
        HiddenFolder.query.filter_by(hidden_by=uid).update({'hidden_by': None})
        db.session.delete(user)
        db.session.commit()
        logger.info(f"Admin '{current_user.username}' deleted user '{deleted_name}'")
        return jsonify({'status': 'ok'})
    except Exception as e:
        db.session.rollback()
        logger.error(f"User delete error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/folders')
@login_required
@admin_required
def admin_get_folders():
    """All folders (including hidden), with hidden status."""
    hidden = {h.folder for h in HiddenFolder.query.all()}
    return jsonify([{'name': f, 'hidden': f in hidden} for f in list_r2_folders()])


@app.route('/api/admin/folders/toggle_hidden', methods=['POST'])
@login_required
@admin_required
def admin_toggle_folder_hidden():
    folder = (request.json or {}).get('folder', '').strip()
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400
    record = HiddenFolder.query.filter_by(folder=folder).first()
    if record:
        db.session.delete(record)
        hidden = False
    else:
        db.session.add(HiddenFolder(folder=folder, hidden_by=current_user.id))
        hidden = True
    db.session.commit()
    return jsonify({'folder': folder, 'hidden': hidden})


@app.route('/api/admin/assignments')
@login_required
@admin_required
def admin_get_assignments():
    folder = request.args.get('folder')
    if not folder:
        return jsonify([])
    rows = db.session.query(Assignment, User).join(
        User, Assignment.user_id == User.id
    ).filter(Assignment.folder == folder).order_by(User.username).all()
    return jsonify([{
        'id': a.id, 'user_id': a.user_id, 'username': u.username,
        'start_index': a.start_index, 'end_index': a.end_index, 'note': a.note or ''
    } for a, u in rows])


@app.route('/api/admin/assignments/save', methods=['POST'])
@login_required
@admin_required
def admin_save_assignment():
    data = request.json or {}
    user_id = data.get('user_id')
    folder = data.get('folder', '').strip()
    start = data.get('start_index')
    end = data.get('end_index')
    note = data.get('note', '')
    if not all([user_id, folder, start is not None, end is not None]):
        return jsonify({'error': 'Missing required fields'}), 400
    if start < 0 or end < start:
        return jsonify({'error': 'Invalid range'}), 400
    try:
        record = Assignment.query.filter_by(user_id=user_id, folder=folder).first()
        if not record:
            record = Assignment(user_id=user_id, folder=folder, created_by=current_user.id)
            db.session.add(record)
        record.start_index = start
        record.end_index = end
        record.note = note or None
        db.session.commit()
        return jsonify({'id': record.id})
    except Exception as e:
        db.session.rollback()
        logger.error(f"Assignment save error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/assignments/<int:aid>', methods=['DELETE'])
@login_required
@admin_required
def admin_delete_assignment(aid):
    record = Assignment.query.get(aid)
    if not record:
        return jsonify({'error': 'Not found'}), 404
    db.session.delete(record)
    db.session.commit()
    return jsonify({'status': 'ok'})


@app.route('/api/assignments/me')
@login_required
def get_my_assignment():
    """Return the current user's assignment for a folder, with resolved image keys."""
    folder = request.args.get('folder')
    if not folder:
        return jsonify({'assignment': None})
    assignment = Assignment.query.filter_by(user_id=current_user.id, folder=folder).first()
    if not assignment:
        return jsonify({'assignment': None})
    try:
        prefix = f"classifier/{folder}/"
        all_keys = sorted(k for k in list_all_image_keys(prefix)
                          if k.lower().endswith(('.png', '.jpg', '.jpeg')))
        if not all_keys:
            prefix = f"{folder}/"
            all_keys = sorted(k for k in list_all_image_keys(prefix)
                              if k.lower().endswith(('.png', '.jpg', '.jpeg')))
        assigned = all_keys[assignment.start_index:assignment.end_index + 1]
    except Exception as e:
        logger.error(f"Error resolving assignment keys: {e}", exc_info=True)
        assigned = []
    return jsonify({'assignment': {
        'id': assignment.id,
        'start_index': assignment.start_index,
        'end_index': assignment.end_index,
        'total': len(assigned),
        'keys': assigned,
    }})


@app.route('/api/dashboard/summary')
@login_required
def dashboard_summary():
    """Aggregate classification stats per folder and user from the DB."""
    records = db.session.query(
        Classification.image_key, Classification.user_id, Classification.category, User.username
    ).join(User, Classification.user_id == User.id).filter(
        Classification.category != None,
        Classification.category != ''
    ).all()

    folder_data: Dict[str, Dict] = {}
    for image_key, user_id, category, username in records:
        parts = image_key.split('/')
        if len(parts) >= 3 and parts[0] == 'classifier':
            folder = parts[1]
        elif len(parts) >= 2:
            folder = parts[0]
        else:
            folder = 'unknown'

        if folder not in folder_data:
            folder_data[folder] = {'users': {}, 'keys': set(), 'key_votes': {}, 'cat_counts': {}}
        fd = folder_data[folder]
        fd['users'][username] = fd['users'].get(username, 0) + 1
        fd['keys'].add(image_key)
        if image_key not in fd['key_votes']:
            fd['key_votes'][image_key] = {}
        fd['key_votes'][image_key][user_id] = category
        fd['cat_counts'][category] = fd['cat_counts'].get(category, 0) + 1

    folders_out = []
    for folder, fd in sorted(folder_data.items()):
        multi = {k: v for k, v in fd['key_votes'].items() if len(v) >= 2}
        agree_rate = (
            sum(1 for v in multi.values() if len(set(v.values())) == 1) / len(multi)
            if multi else None
        )
        folders_out.append({
            'name': folder,
            'users': dict(sorted(fd['users'].items())),
            'total_classified_images': len(fd['keys']),
            'category_counts': fd['cat_counts'],
            'agree_rate': round(agree_rate, 3) if agree_rate is not None else None,
        })

    user_totals: Dict[str, Dict] = {}
    for fd in folder_data.values():
        for uname, cnt in fd['users'].items():
            if uname not in user_totals:
                user_totals[uname] = {'total': 0, 'folders': 0}
            user_totals[uname]['total'] += cnt
            user_totals[uname]['folders'] += 1

    return jsonify({'folders': folders_out, 'user_totals': dict(sorted(user_totals.items()))})


@app.route('/api/bulk_import/preview', methods=['POST'])
@login_required
def bulk_import_preview():
    """Parse an uploaded CSV/FITS file and return a preview of what would be imported."""
    folder = request.form.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file uploaded'}), 400

    fname = (f.filename or '').lower()

    try:
        if fname.endswith('.csv'):
            import csv as csv_mod
            content = f.read().decode('utf-8', errors='replace')
            reader = csv_mod.DictReader(io.StringIO(content))
            raw_rows = list(reader)
        elif fname.endswith(('.fits', '.fit')):
            from astropy.table import Table
            table = Table.read(io.BytesIO(f.read()))
            raw_rows = [
                {col: ('' if _serialize_fits_value(row[col]) is None else str(_serialize_fits_value(row[col])).strip())
                 for col in table.colnames}
                for row in table
            ]
        else:
            return jsonify({'error': 'Unsupported format — please upload a .csv or .fits file'}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {e}'}), 400

    if not raw_rows:
        return jsonify({'error': 'File is empty'}), 400

    cols = list(raw_rows[0].keys())
    id_col = next((c for c in cols if c.upper() == 'ID'), None)
    cat_col = next((c for c in cols if c.lower() == 'category'), None)

    if not id_col:
        return jsonify({'error': f'No ID column found. Columns present: {cols}'}), 400
    if not cat_col:
        return jsonify({'error': f'No category column found. Columns present: {cols}'}), 400

    # List images currently in the folder
    try:
        prefix = f"classifier/{folder}/"
        all_keys = [k for k in list_all_image_keys(prefix) if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not all_keys:
            prefix = f"{folder}/"
            all_keys = [k for k in list_all_image_keys(prefix) if k.lower().endswith(('.png', '.jpg', '.jpeg'))]
    except Exception as e:
        return jsonify({'error': f'Failed to list images in folder: {e}'}), 500

    stem_to_key: Dict[str, str] = {}
    for key in all_keys:
        stem = os.path.splitext(key.split('/')[-1])[0]
        stem_to_key[stem] = key

    # Current user's existing classifications for this folder
    try:
        existing = Classification.query.filter(
            Classification.user_id == current_user.id,
            Classification.category != None,
            Classification.category != '',
            db.or_(
                Classification.image_key.like(f"classifier/{folder}/%"),
                Classification.image_key.like(f"{folder}/%"),
            )
        ).all()
        existing_map = {r.image_key: r.category for r in existing}
    except Exception as e:
        logger.error(f"Error fetching existing classifications: {e}", exc_info=True)
        existing_map = {}

    matched = []
    unmatched = []
    seen_ids: set = set()

    for row in raw_rows:
        row_id = str(row.get(id_col, '') or '').strip()
        category = str(row.get(cat_col, '') or '').strip()
        if not row_id or not category:
            continue
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)

        key = stem_to_key.get(row_id)
        if key is None:
            unmatched.append(row_id)
        else:
            current_cat = existing_map.get(key, '')
            matched.append({
                'id': row_id,
                'key': key,
                'category': category,
                'conflict': bool(current_cat),
                'current_category': current_cat,
            })

    return jsonify({
        'total_file_rows': len(raw_rows),
        'matched': matched,
        'unmatched': unmatched,
    })


@app.route('/api/bulk_import/commit', methods=['POST'])
@login_required
def bulk_import_commit():
    """Apply a previewed bulk import to the database."""
    data = request.json or {}
    conflict_mode = data.get('conflict_mode', 'skip')  # 'overwrite' or 'skip'
    rows = data.get('rows', [])

    imported = 0
    skipped = 0
    errors: List[str] = []

    for row in rows:
        key = row.get('key', '').strip()
        category = row.get('category', '').strip()
        is_conflict = row.get('conflict', False)

        if not key or not category:
            continue
        if is_conflict and conflict_mode == 'skip':
            skipped += 1
            continue

        try:
            record = Classification.query.filter_by(
                user_id=current_user.id, image_key=key
            ).first()
            if not record:
                record = Classification(user_id=current_user.id, image_key=key)
                db.session.add(record)
            record.category = category
            imported += 1
        except Exception as e:
            errors.append(f'{key}: {e}')

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Bulk import commit error: {e}", exc_info=True)
        return jsonify({'error': f'Database commit failed: {e}'}), 500

    logger.info(f"Bulk import by user {current_user.id}: {imported} imported, {skipped} skipped, {len(errors)} errors")
    return jsonify({'imported': imported, 'skipped': skipped, 'errors': errors})


@app.route('/api/compare')
@login_required
def get_compare():
    """Return all users' classifications for a folder as {users, classifications} for the comparison table."""
    folder = request.args.get('folder')
    if not folder:
        return jsonify({'error': 'No folder specified'}), 400

    try:
        users = User.query.order_by(User.username).all()
        records = Classification.query.filter(
            Classification.category != None,
            Classification.category != '',
            db.or_(
                Classification.image_key.like(f"classifier/{folder}/%"),
                Classification.image_key.like(f"{folder}/%"),
            )
        ).all()

        by_key: Dict[str, Dict[str, str]] = {}
        for r in records:
            if r.image_key not in by_key:
                by_key[r.image_key] = {}
            by_key[r.image_key][str(r.user_id)] = r.category

        return jsonify({
            'users': [{'id': str(u.id), 'username': u.username} for u in users],
            'classifications': by_key,
        })
    except Exception as e:
        logger.error(f"Error fetching compare data for folder {folder}: {e}", exc_info=True)
        return jsonify({'error': 'Database error'}), 500


# Create DB tables and run lightweight column migrations.
# Each operation is wrapped independently so a race-condition failure in one
# (multiple Gunicorn workers starting simultaneously) doesn't block the others.
with app.app_context():
    try:
        db.create_all()
    except Exception as e:
        logger.warning(f"DB create_all (likely race condition): {e}")

    try:
        inspector = inspect(db.engine)
        if inspector.has_table("users"):
            cols = {c['name'] for c in inspector.get_columns('users')}
            if 'is_admin' not in cols:
                with db.engine.connect() as conn:
                    # 'false' works in both PostgreSQL and modern SQLite
                    conn.execute(db.text('ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT false'))
                    conn.commit()
                    logger.info("Migrated: added is_admin column to users")
    except Exception as e:
        logger.warning(f"DB migration (is_admin): {e}")

    try:
        if ADMIN_USERNAMES:
            promoted = User.query.filter(
                User.username.in_(ADMIN_USERNAMES), User.is_admin.is_(False)
            ).update({'is_admin': True}, synchronize_session=False)
            db.session.commit()
            if promoted:
                logger.info(f"Promoted {promoted} user(s) to admin via ADMIN_USERNAMES")
    except Exception as e:
        logger.warning(f"ADMIN_USERNAMES promotion: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
