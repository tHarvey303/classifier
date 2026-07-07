# Source Classifier

A self-hosted web app for teams to visually classify large sets of images — built for vetting astronomical source cutouts, but usable for any image-labelling problem (galaxy morphology, artefact rejection, ML training-set curation, etc.).

It is a single-file Flask app backed by Postgres (or SQLite) for classifications and any S3-compatible object store (Cloudflare R2, AWS S3, MinIO, …) for the images themselves.

## Features

- **Multi-user** with password login, an optional registration passphrase, and admin roles
- **Fully keyboard-drivable** classification UI with configurable categories and free-text notes
- **Folder-based datasets**: each prefix in your bucket becomes a selectable dataset, with per-folder category overrides
- **Optional FITS catalog integration**: show catalog metadata next to each image, filter with NumPy-style expressions (e.g. `log10(mass) > 9 & z < 2`)
- **Admin dashboard**: manage users, hide folders, assign index ranges to users, or auto-split a folder across the team with configurable overlap for inter-rater calibration
- **Team dashboard**: per-folder progress, per-user counts, category breakdowns, and agreement rates
- **Export & backup**: CSV export of classifications, bulk import, user-vs-user comparison, and DB backup endpoints

## How it works

- Images live in an object-storage bucket; the app lists them via the S3 API and the browser loads them directly from the bucket's **public domain/CDN** (`R2_PUBLIC_DOMAIN`) — the Flask server never proxies image bytes, so it stays fast and cheap.
- Classifications, users, and assignments live in the database (`DATABASE_URL`).
- Categories come from the local `config.yaml`, unless a folder has its own `config.yaml` in the bucket.

---

## 1. Prepare your images

### Bucket layout

The app looks for "folders" (prefixes) under `classifier/` in your bucket, falling back to the bucket root:

```
your-bucket/
└── classifier/
    ├── survey-A/
    │   ├── 10023.png
    │   ├── 10057.png
    │   ├── catalog.fits      # optional
    │   └── config.yaml       # optional per-folder categories
    └── survey-B/
        └── ...
```

- Supported image formats: `.png`, `.jpg`, `.jpeg`
- **`catalog.fits`** (optional): a FITS table with an `ID` column whose values match the image filenames without extension (e.g. `ID = 10023` ↔ `10023.png`). All other columns are shown as metadata and are available in filter expressions.
- **`config.yaml`** (optional, per folder): overrides the global category list, e.g.

  ```yaml
  categories:
    - Good
    - Bad
    - Unsure
  ```

Upload with any S3 tool, e.g. `rclone`:

```bash
rclone copy ./my_cutouts/ r2:your-bucket/classifier/survey-A/
```

### Where to store the images

Any S3-compatible store works. Options, roughly in order of convenience:

| Provider | Notes |
|---|---|
| **Cloudflare R2** (recommended) | Free egress, 10 GB free storage, easy public custom domains. |
| **AWS S3** | Ubiquitous; egress costs money. Use CloudFront or public-read bucket for `R2_PUBLIC_DOMAIN`. |
| **Backblaze B2** | Cheap storage, S3-compatible API, free egress via Cloudflare. |
| **MinIO** (self-hosted) | Run on the same VPS if your dataset is small; point `R2_ENDPOINT_URL` at it. |

**Cloudflare R2 setup** (the path of least resistance):

1. Cloudflare dashboard → R2 → **Create bucket**.
2. Bucket → Settings → **Public access**: connect a custom domain (e.g. `cdn.example.com`) or enable the `*.r2.dev` development URL. This becomes `R2_PUBLIC_DOMAIN`.
3. R2 → **Manage API Tokens** → create a token with *Object Read & Write* scoped to the bucket. This gives you `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.
4. The S3 endpoint is `https://<ACCOUNT_ID>.r2.cloudflarestorage.com` → `R2_ENDPOINT_URL`.

> The images are publicly readable at `R2_PUBLIC_DOMAIN` — only classify data you're comfortable serving from an (unlisted) public URL. If you need private images, put the CDN behind an auth proxy or add presigned-URL support.

## 2. Configure your environment

```bash
cp .env.example .env
```

Then edit `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `FLASK_SECRET_KEY` | yes | Session signing. Generate: `python -c "import secrets; print(secrets.token_hex(24))"` |
| `DATABASE_URL` | no | SQLAlchemy URL. Defaults to `sqlite:///local.db`; the docker-compose stack uses the bundled Postgres. |
| `POSTGRES_PASSWORD` | compose only | Password for the bundled Postgres container; must match `DATABASE_URL`. |
| `R2_ENDPOINT_URL` | yes | S3 API endpoint of your store. |
| `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` | yes | Credentials with read access to the bucket. |
| `R2_BUCKET_NAME` | yes | Bucket containing the `classifier/` prefix. |
| `R2_PUBLIC_DOMAIN` | yes | Public hostname the browser loads images from. |
| `REGISTRATION_PASSPHRASE` | no | Shared key required to register; unset = open registration. |
| `ADMIN_USERNAMES` | no | Comma-separated usernames auto-granted admin on login. |
| `CLASSIFIER_DOMAIN` | compose only | Hostname Traefik routes to the app. |

**Never commit `.env`** — it is in `.gitignore`. Only `.env.example` (placeholders) belongs in the repo.

## 3. Run it

### Locally (quick test)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py            # or: gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Without `DATABASE_URL` set it uses a local SQLite file — fine for trying it out or a single-user project. Tables are created automatically on first run. Visit `http://localhost:5000`, register (with the passphrase, if set), and start classifying.

### Docker (single container)

```bash
docker build -t classifier .
docker run --env-file .env -p 5000:5000 classifier
```

### Docker Compose (app + Postgres)

`docker-compose.yml` runs the app under Gunicorn plus a Postgres 15 container with a persistent volume:

```bash
docker compose up -d --build
docker compose logs -f web
```

The compose file is written for a [Traefik](https://traefik.io) reverse proxy (see below). If you don't use Traefik, delete the `labels:` and `traefik_network` entries and re-enable the `ports:` mapping to expose port 5000 directly (then put nginx/Caddy in front for TLS).

## 4. Deploy on a VPS

Any small VPS works (1 vCPU / 1 GB RAM is plenty — images are served by the CDN, not the app). Hetzner, DigitalOcean, Linode, etc.

1. **Provision** an Ubuntu/Debian VPS, point a DNS `A` record (e.g. `classifier.example.com`) at it, and install Docker:
   ```bash
   curl -fsSL https://get.docker.com | sh
   ```
2. **Clone and configure**:
   ```bash
   git clone https://github.com/<you>/<repo>.git && cd <repo>
   cp .env.example .env && nano .env    # fill in secrets; set CLASSIFIER_DOMAIN
   ```
3. **Reverse proxy.** The compose file expects a Traefik instance with:
   - an external Docker network called `traefik_network`,
   - a `websecure` (443) entrypoint,
   - a certificate resolver named `letsencrypt`.

   If your Traefik setup names things differently, edit the labels in `docker-compose.yml`. No Traefik yet? Create the network (`docker network create traefik_network`) and run a standard Traefik container attached to it, or strip the Traefik bits and use Caddy:
   ```
   # Caddyfile — Caddy handles TLS automatically
   classifier.example.com {
       reverse_proxy localhost:5000
   }
   ```
4. **Launch**: `docker compose up -d --build`, then open `https://classifier.example.com`, register your admin account (a username listed in `ADMIN_USERNAMES` becomes admin automatically), and hand teammates the URL + registration passphrase.

**Backups**: classifications are the valuable part. Dump the DB periodically, e.g.

```bash
docker compose exec db pg_dump -U postgres classifier > backup_$(date +%F).sql
```

or use the in-app `/api/db_backup` / CSV export endpoints (admin only).

## 5. Adapt it to your own problem

1. **Categories** — edit `config.yaml` (global default) and/or drop a `config.yaml` into each bucket folder. No code changes needed.
2. **Images** — render whatever your "source" is (cutout, spectrum, light curve, photo…) to PNG/JPEG and upload one file per object.
3. **Metadata** — optional: build a `catalog.fits` with an `ID` column matching your filenames plus whatever columns help classifiers decide. The filter box accepts expressions like `snr > 5 & log10(flux) < -18`.
4. **Team workflow** — as admin, use *Assignments* to give each person an index range, or *Auto-split* to divide a folder with N% overlap so you can measure inter-classifier agreement on the dashboard.
5. **Results** — export per-folder CSVs from the app, or query the `classification` table directly (`user_id`, `image_key`, `category`, `notes`, `timestamp`).

## Repository layout

```
app.py              # the whole app: models, API, and inline HTML templates
config.yaml         # default classification categories
requirements.txt
Dockerfile          # python:3.11-slim + gunicorn
docker-compose.yml  # web + postgres, Traefik-ready
.env.example        # template for your secrets (copy to .env)
```
