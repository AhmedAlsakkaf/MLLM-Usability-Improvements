# Deploying the UI Analysis Lab

The app runs live at a subdomain such as `https://uilab.quvoprep.com`.

## What each service does

| Service    | Role |
|------------|------|
| **Contabo**    | The server itself (a VPS: a rented Linux computer with a public IP address). |
| **Coolify**    | A control panel installed on that server. It pulls this repo from GitHub, builds the `Dockerfile`, runs the container, and sets up HTTPS. |
| **Cloudflare** | Manages DNS for `quvoprep.com`. You add one record that points the subdomain at the Contabo server. |

Request flow: browser → Cloudflare DNS → Contabo server → Coolify proxy → app container (port 8501).

## 1. Find the server IP

Contabo customer panel → **Your services / VPS** → copy the **IPv4 address**.
(Coolify also shows it under **Servers**.)

## 2. Create the subdomain in Cloudflare

Cloudflare dashboard → `quvoprep.com` → **DNS → Records → Add record**:

| Type | Name    | Content (IPv4)       | Proxy status |
|------|---------|----------------------|--------------|
| A    | `uilab` | `<Contabo server IP>`| **DNS only** (grey cloud) |

Keep it **DNS only** at first so Coolify can obtain its Let's Encrypt certificate.
If you later switch to **Proxied** (orange cloud), set **SSL/TLS → Overview → Full (strict)**,
or the site will loop or show SSL errors.

## 3. Create the app in Coolify

1. Open Coolify (usually `http://<server IP>:8000`, or the domain your friend set up) and log in.
2. **Projects → + Add** (e.g. "Research"), open it, then **+ New → Resource**.
3. Pick **Public Repository** and paste
   `https://github.com/AIG-ist-tugraz/MLLM-Usability-Improvements`.
   (If the repo is private, use **Private Repository (with GitHub App)** and connect GitHub first.)
4. Settings:
   - **Branch:** `ahmed` (or `main` once merged)
   - **Build Pack:** `Dockerfile`
   - **Ports Exposes:** `8501`
5. **Domains:** `https://uilab.quvoprep.com`
6. **Environment Variables** (tab in the resource):

   | Key              | Value |
   |------------------|-------|
   | `GEMINI_API_KEY` | your Google AI Studio key |
   | `MLLM_MODEL`     | `gemini-3.8-flash` |
   | `APP_PASSWORD`   | a password to share with your users |

   `APP_PASSWORD` matters. Without it, anyone who finds the URL can use your Gemini quota.
7. *(Optional, to keep results across redeploys)* **Persistent Storage → + Add → Volume**,
   destination path `/app/data`.
8. Click **Deploy** and watch the build log. The first build takes a few minutes because it installs torch.
9. Open `https://uilab.quvoprep.com`.

## Updating

Push to the deployed branch, then click **Redeploy** in Coolify, or turn on
**Auto Deploy** (needs the GitHub App or a webhook).

## Troubleshooting

- **Site not reachable:** check that the DNS record points to the right IP (`nslookup uilab.quvoprep.com`)
  and that the Contabo firewall allows ports 80 and 443.
- **"No GEMINI_API_KEY":** the environment variable is missing in Coolify; add it, then redeploy.
- **Page loads but stays on "Please wait…":** if Cloudflare is proxied, check that SSL mode is Full (strict).
- **Running locally:** nothing changes. Use `start_app.bat` as before (no password locally).
