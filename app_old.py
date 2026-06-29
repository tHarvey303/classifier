import os
import sys
import logging
import yaml
import boto3
from sqlalchemy import inspect
from dotenv import load_dotenv
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from botocore.client import Config
from typing import List, Dict, Optional

# --- Load Environment Variables from .env ---
load_dotenv()

# --- Logging Configuration ---
# This ensures logs are printed to the console (stdout) so Docker can capture them.
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

# Registration Security
REGISTRATION_PASSPHRASE = os.environ.get('REGISTRATION_PASSPHRASE')

# R2 / S3 Config
R2_ENDPOINT = os.environ.get('R2_ENDPOINT_URL', '')
R2_KEY = os.environ.get('R2_ACCESS_KEY_ID')
R2_SECRET = os.environ.get('R2_SECRET_ACCESS_KEY')
BUCKET_NAME = os.environ.get('R2_BUCKET_NAME')
R2_PUBLIC_DOMAIN = os.environ.get('R2_PUBLIC_DOMAIN', 'cdn.tharvey.space')

# FIX: Auto-correct endpoint if https is missing
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

# --- Database Models ---
class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class Classification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    image_key = db.Column(db.String(500), nullable=False)
    category = db.Column(db.String(100))
    notes = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, server_default=db.func.now(), onupdate=db.func.now())

    __table_args__ = (db.UniqueConstraint('user_id', 'image_key', name='_user_image_uc'),)

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
            Existing users: Enter credentials.<br>
            New users: Enter a username, password, and the Registration Key.
        </div>
    </div>
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
        
        .btn-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1.5rem; }
        .btn-cat { padding: 0.75rem; border: 1px solid #d1d5db; background: white; border-radius: 0.5rem; cursor: pointer; text-align: left; color: var(--text-main); transition: all 0.2s; }
        .btn-cat:hover { background: #f3f4f6; border-color: #9ca3af; }
        .btn-cat.active { background: var(--accent); color: white; border-color: var(--accent); }
        
        .notes-area { width: 100%; height: 100px; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 0.5rem; resize: vertical; margin-bottom: 1.5rem; box-sizing: border-box; font-family: inherit; }
        
        .nav-row { display: flex; gap: 1rem; margin-top: auto; }
        .btn-nav { flex: 1; padding: 0.75rem; background: #1f2937; color: white; border: none; border-radius: 0.5rem; cursor: pointer; font-weight: 600; }
        .btn-nav:hover { background: #374151; }

        #toast { position: fixed; top: 20px; right: 20px; background: #10b981; color: white; padding: 10px 20px; border-radius: 5px; opacity: 0; transition: opacity 0.3s; pointer-events: none; z-index: 99; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
</head>
<body>

<div class="header">
    <h1>Source Classifier</h1>
    <div class="user-info">
        <b>{{ current_user.username }}</b>
        <a href="{{ url_for('logout') }}">Logout</a>
    </div>
</div>

<div class="container">
    <div class="image-area">
        <img id="main-image" src="" alt="Select a folder to begin">
    </div>
    
    <div class="sidebar">
        <h2 id="filename-display">Welcome</h2>
        <div class="meta" id="counter-display">Select a folder</div>

        <label class="section-label">Folder</label>
        <select id="folder-select" class="dropdown" onchange="loadFolder()">
            <option value="" disabled selected>Loading folders...</option>
        </select>

        <label class="section-label">Filter Status</label>
        <select id="filter-select" class="dropdown" onchange="fetchImages()">
            <option value="All">All Images</option>
            <option value="Uncategorized">Uncategorized</option>
        </select>

        <label class="section-label">Classification</label>
        <div id="category-container" class="btn-grid"></div>

        <label class="section-label">Notes</label>
        <textarea id="notes-input" class="notes-area" placeholder="Private notes..."></textarea>

        <div class="nav-row">
            <button class="btn-nav" onclick="navigate(-1)">Previous</button>
            <button class="btn-nav" onclick="navigate(1)">Next</button>
        </div>
    </div>
</div>
<div id="toast">Saved</div>

<script>
    let images = [];
    let currentIndex = 0;
    let categories = [];
    let currentFolder = "";

    async function init() {
        // 1. Load Config (Categories)
        const configRes = await fetch('/api/config');
        const configData = await configRes.json();
        categories = configData.categories;
        
        // Render Category Buttons
        const container = document.getElementById('category-container'); 
        categories.forEach((cat, index) => {
            const btn = document.createElement('button');
            btn.className = 'btn-cat';
            // Show Shortcut in label if within 1-9
            if (index < 9) {
                btn.innerText = `${index + 1}. ${cat}`;
            } else {
                btn.innerText = cat;
            }
            
            btn.onclick = () => selectCategory(cat);
            btn.dataset.category = cat;
            container.appendChild(btn);
        });

        // Render Filter Options
        const filterSelect = document.getElementById('filter-select');
        categories.forEach(cat => {
            const opt = document.createElement('option');
            opt.value = cat;
            opt.innerText = cat;
            filterSelect.appendChild(opt);
        });

        // 2. Fetch Folders
        await fetchFolders();

        // 3. Setup Shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'TEXTAREA') return;
            
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

    async function loadFolder() {
        currentFolder = document.getElementById('folder-select').value;
        await fetchImages();
    }

    async function fetchImages() {
        if (!currentFolder) return;
        
        const filterVal = document.getElementById('filter-select').value;
        const res = await fetch(`/api/images?folder=${encodeURIComponent(currentFolder)}&filter=${encodeURIComponent(filterVal)}`);
        images = await res.json();
        
        if (images.length > 0) {
            loadStateForImage(0);
        } else {
            document.getElementById('filename-display').innerText = "No images found";
            document.getElementById('main-image').src = "";
            document.getElementById('counter-display').innerText = "0 / 0";
            currentIndex = -1;
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
    }

    async function selectCategory(cat) {
        highlightCategory(cat);
        await saveCurrentState();
    }

    async function navigate(dir) {
        if (currentIndex === -1) return;
        await saveCurrentState();
        let newIndex = currentIndex + dir;
        if (newIndex >= 0 && newIndex < images.length) {
            loadStateForImage(newIndex);
        }
    }

    init();
</script>
</body>
</html>
"""

# --- Routes ---

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
                # Create Account
                new_user = User(username=username, password_hash=generate_password_hash(password))
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
    return render_template_string(APP_TEMPLATE)

@app.route('/api/config')
@login_required
def get_config():
    if os.path.exists("config.yaml"):
        with open("config.yaml", 'r') as f:
            return jsonify(yaml.safe_load(f))
    return jsonify({"categories": ["Good", "Bad"]})

@app.route('/api/folders')
@login_required
def get_folders():
    """Lists subdirectories under classifier/"""
    logger.info("Request: /api/folders started")
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
    logger.info(f"Returning folders: {folders}")
    return jsonify(folders)

@app.route('/api/images')
@login_required
def get_images():
    folder = request.args.get('folder')
    logger.info(f"Request: /api/images started for folder: {folder}")
    if not folder:
        return jsonify([])

    try:
        # Try specific classifier prefix first
        prefix = f"classifier/{folder}/"
        response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
        
        # If empty, maybe the folder is at root?
        if not response.get('Contents'):
             logger.info(f"No contents in {prefix}, trying root prefix {folder}/")
             prefix = f"{folder}/"
             response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        all_keys = [
            obj['Key'] for obj in response.get('Contents', []) 
            if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        all_keys.sort()
        logger.info(f"Found {len(all_keys)} images")
    except Exception as e:
        logger.error(f"Error listing images: {e}", exc_info=True)
        return jsonify([])

    # Apply Classification Filters
    filter_val = request.args.get('filter', 'All')
    
    if filter_val == 'All':
        return jsonify(all_keys)

    try:
        user_classes = Classification.query.filter_by(user_id=current_user.id).all()
        class_map = {c.image_key: c.category for c in user_classes}
    except Exception as e:
        logger.error(f"Error fetching user classifications: {e}", exc_info=True)
        class_map = {}

    filtered = []
    for key in all_keys:
        cat = class_map.get(key, "")
        if filter_val == 'Uncategorized':
            if not cat: filtered.append(key)
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

# Create DB tables if they don't exist
with app.app_context():
    try:
        inspector = inspect(db.engine)
        if not inspector.has_table("users"):
            db.create_all()
    except Exception as e:
        # Catch race conditions where another worker creates the table concurrently
        logger.warning(f"DB Init ignored (likely race condition): {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
