# Telegram ChatGPT Bot with Telegram Stars Billing

Project link: [@tg_own_gpt_bot](https://t.me/tg_own_gpt_bot)

This project is a Telegram bot that acts as a "bridge" between Telegram and OpenAI (ChatGPT)
using the **OpenAI Responses API**. The bot supports text, images, files, chat history,
and a paid access model for regular users via **Telegram Stars**.

Administrators listed in the `ADMINS` environment variable can use the bot for free.
All other users pay per message in Telegram Stars.

---

## Features

- ChatGPT integration via the OpenAI Responses API.
- Multiple model support (e.g., `gpt-5`, `gpt-5.1`, `gpt-4o`, `gpt-5-nano`, etc.).
- Handles **text, photos, and documents (including PDF)**.
- Text file contents are embedded into the context; PDF files are uploaded to OpenAI.
- Conversation history is stored in SQLite and can be exported to PDF.
- Dialogue modes:
  - **continuous** - one long conversation with history;
  - **one-shot** - a new conversation for every message.
- Billing via **Telegram Stars**:
  - 1 free message for every new user;
  - each further message costs a fixed number of stars;
  - balance top-up via Telegram Payments / Stars invoices (currency `XTR`);
  - balances and payments are stored in the database.
- Admins (by Telegram ID) are always free, no stars are charged.
- `/balance` and `/topup` commands to manage user balance.

---

## Requirements

- Python 3.10+
- Telegram account and a bot created via @BotFather.
- Bot token (Bot Token).
- OpenAI account and API key with access to the models you want to use.
- Installed dependencies from `requirements.txt` (example below).
- SQLite (comes with Python).

Example `requirements.txt` (approximate):

```text
python-telegram-bot~=21.0
openai~=1.0
reportlab~=4.0
```

(Exact versions may differ depending on the libraries you use.)

---

## Configuration and Environment Variables

The bot is configured via environment variables. You can export them in your shell
or put them into a `.env` file and load with your preferred method.

### Required variables

- `TELEGRAM_BOT_TOKEN` - Telegram bot token from @BotFather.
- `OPENAI_API_KEY` - OpenAI API key.
- `DB_PATH` - path to the SQLite database file (for example `./bot.db`).

### Users and access control

- `ADMINS` - comma-separated list of Telegram user IDs that are treated as admins.
  Admins never pay for messages. Example:

  ```env
  ADMINS=123456789,987654321
  ```

- `ALLOWED_USERS` - (optional) comma-separated list of Telegram IDs that are explicitly
  allowed to use the bot. This is only used if strict allowlist mode is enabled.

- `RESTRICT_TO_ALLOWLIST` - if set to `true` / `1`, the bot is only available to users from
  `ALLOWED_USERS` (admins always have access regardless).  
  If **not set**, the bot is open to everyone.

### Billing settings

- `MESSAGE_PRICE_STARS` - price of **one message** for regular users in Telegram Stars.

  ```env
  MESSAGE_PRICE_STARS=2
  ```

  This means each paid message costs `2⭐`.

- Top-up packages (10, 20, 50, 100, 200, 500, 1000⭐) are defined in the code as the constant
  `TOPUP_OPTIONS`.

### Other optional settings

- You may have extra tuning variables such as:
  - `MAX_HISTORY_MSGS` - max number of messages kept in the in-context history;
  - `MAX_OUTPUT_TOKENS` - maximum number of tokens for model output;
  - `CONTEXT_CHAR_BUDGET` - character limit for the prompt context;
  - etc., depending on what is defined in `tg_chatgpt_bot.py`.

---

## Database structure

The SQLite database is created automatically on first run and includes at least the following tables:

- `chat_settings` - per-chat settings (selected model, dialogue mode).
- `messages` - conversation history (role: system/user/assistant, content, timestamp).
- `users` - user records with their balance and free-message state:
  - `user_id` - Telegram user ID;
  - `balance_stars` - current balance in stars;
  - `free_message_used` - 0/1, whether the free message was already used.

- `payments` - recorded top-ups via Telegram Stars:
  - `user_id` - Telegram user ID;
  - `stars_amount` - number of stars added;
  - `telegram_payment_charge_id`, `provider_payment_charge_id` - payment identifiers;
  - `invoice_payload` - invoice payload string.

---

## Billing logic

1. On the **first message** of a user:
   - a row in `users` is created if it does not exist yet;
   - the user gets **1 free message**;
   - `free_message_used` is set to 1, and no stars are charged.

2. On **subsequent messages**:
   - if the user is an **admin**, no stars are charged at all;
   - if the user is **not an admin**:
     - the bot checks the balance;
     - if `balance_stars < MESSAGE_PRICE_STARS`, the bot offers a top-up
       and does **not** send a request to OpenAI;
     - if the balance is sufficient, the bot **first sends the request to OpenAI** and,
       only after a successful response is generated and sent to the user,
       it **deducts stars** from the balance.

3. Top-up flow:
   - The user calls `/topup` or automatically gets a top-up keyboard when the balance is insufficient;
   - The bot sends a Telegram invoice with `XTR` currency and the chosen stars amount;
   - After successful payment:
     - stars are added to `balance_stars`;
     - a row is inserted into `payments` with all payment details;
     - the user receives a confirmation message and a hint to check `/balance`.

---

## Bot commands

Main commands:

- `/start` - greeting, information about the current model and dialogue mode,
  plus a short description of billing and available commands.

- `/model` - show a list of preset models with inline buttons
  (e.g. gpt-5, gpt-5.1, gpt-4o, gpt-5-nano, etc.).

- `/setmodel <id>` - manually set any OpenAI model ID (must be a valid model on your account).

- `/mode` - toggle dialog mode:
  - **continuous** - keep history;
  - **one-shot** - clear history for each message.

- `/new` - clear current chat history (start from scratch).

- `/export` - export the conversation to a PDF file and send it to the user.

- `/balance` - show the current star balance and remind the user of the message price.

- `/topup` - open the top-up menu (inline keyboard with 10, 20, 50, 100, 200, 500, 1000⭐ packages).

---

## Message handling

### Text messages

Plain text messages are processed by `handle_message`:

1. The bot checks access via `is_allowed_user`.
2. For non-admins:
   - `ensure_balance_for_message` checks if the free message is still available
     or if there is enough balance to pay for the message;
   - if there is not enough balance, a top-up keyboard is shown and processing stops;
   - if everything is fine, the message is turned into `user_parts` and sent to OpenAI.
3. `user_parts` is built from text and any (single) attachment if present.
4. Conversation history is loaded from the database.
5. `run_model_with_tools` is called to get a response from the model.
6. The user receives a reply (plain text or PDF). If everything succeeded and the message
   should be charged, stars are deducted and the answer is stored in history.

During long processing the bot can show a temporary “🤖 Thinking…” status message
(using the `temp_status` context manager).

### Messages with files / media groups

Messages with attachments (photos, multiple files, albums) are processed by `handle_media_group`:

- The bot collects text from captions, photos, and documents inside the media group.
- For image files, the raw bytes are passed as image parts to the model.
- For text documents, their decoded content is split into chunks and added as extra text parts.
- For PDFs, the bot uploads them to OpenAI with `client.files.create` and then passes
  the `file_id` to the model as an `input_file` part.
- Then the logic is similar to `handle_message`: history is loaded, the model is called,
  and the reply is sent back to the user.

A temporary "🤖 Thinking…" message is also shown while the model is processing
and removed afterwards.

---

## Running the bot

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Set environment variables (e.g. in your shell or `.env`):

   ```bash
   export TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
   export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
   export DB_PATH="./bot.db"
   export ADMINS="123456789"
   export MESSAGE_PRICE_STARS="2"
   ```

3. Run the bot:

   ```bash
   python tg_chatgpt_bot.py
   ```

4. Open your bot in Telegram and send `/start`.

---

## Models and web search

The code can support multiple OpenAI models. Typical usage:

- For "larger" models (e.g. `gpt-5`, `gpt-5.1`, `gpt-4o`) you may enable
  tools such as web search (`web_search`) or URL loading (`open_url`).
- For smaller "nano" models it is often more practical to **disable tools**
  and use them as a cheap offline chat (otherwise they may get stuck
  in tool-calling loops).

This behavior is configured inside `run_model_with_tools` by deciding,
for each model, whether to pass `tools=TOOLS` into `client.responses.create`.

---

## License

MIT License
