# Deployment Guide - Hugging Face Spaces

This guide explains how to deploy the GPU Memory Calculator to Hugging Face Spaces.

## Prerequisites

- Hugging Face account (free at [huggingface.co](https://huggingface.co))
- Git installed locally
- Project files ready

## Quick Start

### Option 1: Connect GitHub Repository (Recommended)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Configure your Space:
   - **Owner**: Your username or organization
   - **Space name**: `gpu-memory-calculator` (or your choice)
   - **License**: MIT
   - **SDK**: Docker
   - **Hardware**: CPU basic (free tier)
4. Click **"Create Space"**
5. In the Space settings, click **"Files"** → **"Connect GitHub repository"**
6. Select your GPU Memory Calculator repository
7. Push the Space README to your repo's main branch as `README.md` or use the one in `huggingface_space/README.md`

### Option 2: Git CLI Deployment

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Clone the Space repository:
```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/gpu-memory-calculator
cd gpu-memory-calculator
```

3. Copy project files to the Space:
```bash
# From your project root
cp -r src/ cli/ web/ YOUR_USERNAME/gpu-memory-calculator/
cp Dockerfile requirements.txt .dockerignore YOUR_USERNAME/gpu-memory-calculator/
cp huggingface_space/README.md YOUR_USERNAME/gpu-memory-calculator/README.md
```

4. Commit and push:
```bash
cd YOUR_USERNAME/gpu-memory-calculator
git add .
git commit -m "Initial deployment"
git push
```

The Space will automatically build and deploy upon receiving the push.

## File Structure

Your Space repository should contain:

```
gpu-memory-calculator/
├── Dockerfile              # Container configuration
├── requirements.txt        # Python dependencies
├── .dockerignore          # Files to exclude from Docker image
├── README.md              # Space metadata (YAML frontmatter + description)
├── src/                   # Python package
│   └── gpu_mem_calculator/
├── web/                   # FastAPI web application
│   ├── app.py
│   ├── templates/
│   └── static/
└── cli/                   # Command-line interface (optional)
```

## Port Configuration

Hugging Face Spaces uses **port 7860** by default. The Dockerfile is already configured to expose this port:

```dockerfile
EXPOSE 7860
CMD ["uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Environment Variables

The Dockerfile sets the following environment variables:

- `PORT=7860`: Hugging Face Spaces default port
- `PYTHONUNBUFFERED=1`: Prevent Python output buffering
- `PYTHONDONTWRITEBYTECODE=1`: Skip .pyc file generation

## Hardware Options

| Hardware | vRAM | Cost | Use Case |
|----------|------|------|----------|
| **CPU basic** | N/A | Free | Recommended for this calculator (no model training) |
| **T4 small** | 16GB | ~$0.10/hr | Overkill for this app |
| **T4 medium** | 16GB | ~$0.20/hr | Not needed |
| **L4** | 24GB | ~$0.40/hr | Not needed |
| **A10G** | 24GB | ~$1.00/hr | Not needed |
| **A100** | 80GB | ~$1.60/hr | Not needed |

**Recommendation**: Use the **free CPU basic** tier. The calculator doesn't run actual models, it only estimates memory requirements.

## Verification

After deployment, verify your Space is running:

1. Visit your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/gpu-memory-calculator`
2. Check the status indicator (should be ● Running)
3. Test the calculator:
   - Select a preset model (e.g., "LLaMA 2 7B")
   - Click "Calculate"
   - Verify memory results appear

## Troubleshooting

### Build Fails

1. Check the **"Logs"** tab in your Space
2. Common issues:
   - Missing `requirements.txt`
   - Incorrect Dockerfile syntax
   - Missing project files

### App Won't Start

1. Check **"Runtime"** logs for errors
2. Verify port 7860 is exposed in Dockerfile
3. Ensure `web/app.py` exists and is importable

### "Application not responding"

1. Check if the container is running (● green indicator)
2. Review container logs for Python errors
3. Verify FastAPI is starting: `uvicorn web.app:app ...`

## Space README Frontmatter

The `README.md` must contain YAML frontmatter for Hugging Face to recognize it:

```yaml
---
title: GPU Memory Calculator
emoji: 🎮
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---
```

## Updating the Deployment

To update your deployed Space:

### With GitHub Connected
Simply push to your linked branch. Hugging Face will automatically rebuild.

### With Git CLI
```bash
cd gpu-memory-calculator
# Make changes...
git add .
git commit -m "Update description"
git push
```

## Custom Domain (Optional)

For a custom domain:
1. Go to Space **Settings** → **Custom Domain**
2. Add your domain (e.g., `gpu-calc.yourdomain.com`)
3. Configure DNS CNAME record to `huggingface.co`

## Monitoring

- **Status**: Check the green/red indicator on your Space page
- **Logs**: View real-time logs in the **"Logs"** tab
- **Metrics**: See CPU/memory usage in **"Settings"** → **"Metrics"**

## Security

- The Space is **public** by default
- For private Spaces, upgrade to a paid plan
- No authentication is currently implemented
- Consider adding rate limiting for production use

## Next Steps

1. Share your Space link in the project README
2. Add a badge to your GitHub repository:
```markdown
[![Hugging Face Spaces](https://img.shields.io/badge/🤗-Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/YOUR_USERNAME/gpu-memory-calculator)
```
3. Monitor logs and user feedback
4. Iterate based on usage patterns

## Alternative Deployment Platforms

If Hugging Face Spaces doesn't meet your needs:

| Platform | Free Tier | Key Features |
|----------|-----------|--------------|
| **Railway** | $5 credit | Easy GitHub integration |
| **Render** | 750 hours/month | Free SSL, auto-deploy |
| **Fly.io** | 3 VMs | Global edge deployment |
| **DigitalOcean App Platform** | 750 hours | Simple, scalable |
| **AWS Elastic Beanstalk** | 12 months | Full AWS integration |

See [DEPLOYMENT_OTHER.md](DEPLOYMENT_OTHER.md) for guides on alternative platforms.
