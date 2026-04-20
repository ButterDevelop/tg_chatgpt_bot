# Telegram ChatGPT Bot with Stars Billing & CI/CD

Project link: [@tg_own_gpt_bot](https://t.me/tg_own_gpt_bot)

A powerful, production-ready Telegram bot that bridges Telegram with OpenAI. Features include multi-model support, image/file processing, paid access via Telegram Stars, and a robust CI/CD pipeline.

---

## 🚀 Key Features

- **ChatGPT Integration**: Support for all modern OpenAI models (o1, gpt-4o, etc.).
- **Rich Media Support**: Handles text, photos, documents, and **PDFs** (with OpenAI File upload).
- **Billing System**: Integrated **Telegram Stars** (XTR) payments for regular users.
- **Usage Tracking**: Accurate token counting using `tiktoken` and real-time USD cost calculation.
- **Admin Dashboard**: Enhanced `/admin` command with detailed usage stats and financial monitoring.
- **Dockerized**: Easy deployment with Docker and Docker Compose.
- **CI/CD Built-in**: Automatic builds and deployment using GitHub Actions and GHCR.
- **Robust Config**: Smart environment variable parsing (ignores inline comments).

---

## 🛠 Features in Detail

### 📊 Admin and Monitoring
The bot includes a redesigned `/admin` panel that provides:
- Total registered users and message counts.
- **Usage Statistics**: Input/Output tokens tracked per message.
- **Cost Estimation**: Live USD spending calculation based on current OpenAI pricing (Today, Month, Total).

### 💳 Billing Logic
1. **New Users**: Get 1 free message automatically.
2. **Paid Access**: Subsequent messages cost a configurable number of stars (e.g., 2⭐).
3. **Top-ups**: Built-in `/topup` command using Telegram Stars invoices.
4. **Admins**: Always have free unlimited access.

### 📄 Document Handling
- **Text Files**: Embedded directly into the context.
- **PDF Files**: Processed via OpenAI's File API for high accuracy.
- **Export**: Export any conversation history to a beautifully formatted PDF (using WeasyPrint).

---

## 📦 Getting Started (Docker)

The fastest way to run the bot is using Docker.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ButterDevelop/tg_chatgpt_bot.git
   cd tg_chatgpt_bot
   ```

2. **Configure environment**:
   Copy `.env.example` to `.env` and fill in your tokens.
   ```bash
   cp .env.example .env
   ```

3. **Run with Docker Compose**:
   ```bash
   docker compose up -d
   ```

The bot will be available, and your database will be safe in the `./data` volume.

---

## ☁️ Deployment (CI/CD)

This project is configured for automated deployment via GitHub Actions.

### Setup GitHub Secrets
To enable automated deployment, add these secrets to your GitHub repository:
- `SSH_HOST`: Your server IP.
- `SSH_USER`: SSH username (e.g., `ubuntu`).
- `SSH_KEY`: Your private SSH key.
- `GHCR_TOKEN`: GitHub Personal Access Token (PAT) with `read:packages` scope.
- `PROJECT_PATH`: Absolute path on the server where the bot lives.
- **Sensitive Config**: `TELEGRAM_BOT_TOKEN`, `OPENAI_API_KEY`, `ADMIN_IDS`, `TAVILY_API_KEYS`.

Every time you `git push` to the `main` branch, the bot is automatically built, pushed to GitHub Container Registry, and redeployed on your server.

---

## ⚙️ Configuration (.env)

| Variable | Description |
| :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | Your bot token from @BotFather. |
| `OPENAI_API_KEY` | OpenAI API key. |
| `ADMIN_IDS` | Comma-separated Telegram IDs for admins. |
| `MESSAGE_PRICE_STARS`| How many stars to charge per message. |
| `DB_PATH` | Path to SQLite DB (default: `/app/data/chatgpt_tg.db`). |
| `DEFAULT_MODEL` | Default OpenAI model to use. |

---

## 🐳 Resource Management
The bot is optimized for low-resource environments:
- **CPU Limit**: 0.5 cores
- **Memory Limit**: 300MB
- **Database**: SQLite (persistent via Docker volumes).

---

## 📜 License
MIT License. Created by [ButterDevelop](https://github.com/ButterDevelop).
