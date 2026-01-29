# Streamlit Setup Guide — Investment Research Platform

A beginner-friendly, step-by-step guide to getting the Investment Research Platform running locally with Streamlit.

---

## Prerequisites

Before you begin, make sure you have the following installed on your computer:

- **Python 3.9 or later** — check by running `python --version` or `python3 --version`
- **pip** — the Python package manager (comes with Python)
- **Git** — only needed if you are cloning the repository

If you do not have Python installed, download it from [python.org](https://www.python.org/downloads/) and follow the installer instructions. Make sure to check "Add Python to PATH" during installation on Windows.

---

## Step 1: Get the Project Files

If you already have the project folder, skip to Step 2.

Clone the repository:

```bash
git clone <repository-url>
cd Investment_research_platform
```

---

## Step 2: Create a Virtual Environment (Recommended)

A virtual environment keeps this project's dependencies isolated from the rest of your system.

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

After activation you should see `(venv)` at the beginning of your terminal prompt. Every time you open a new terminal to work on this project, re-run the `activate` command.

---

## Step 3: Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

This installs the following key packages:

| Package | Purpose |
|---------|---------|
| `streamlit` | Web application framework (the dashboard UI) |
| `plotly` | Interactive charts and visualizations |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | Machine learning models |
| `torch` | PyTorch for the LSTM neural network |
| `requests` | Fetching data from Yahoo Finance |
| `rich` | CLI formatting (used by `main.py`) |
| `fredapi` | FRED economic data (optional) |

If the install fails on `torch`, you can install a CPU-only version to save disk space:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Step 4: (Optional) Set Up a FRED API Key

The FRED economic data features are optional. If you want to use them:

1. Create a free account at [https://fred.stlouisfed.org](https://fred.stlouisfed.org)
2. Request an API key from [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
3. Set the environment variable:

**macOS / Linux:**
```bash
export FRED_API_KEY=your_key_here
```

**Windows (Command Prompt):**
```cmd
set FRED_API_KEY=your_key_here
```

To make this permanent, add the export line to your `~/.bashrc`, `~/.zshrc`, or system environment variables on Windows.

---

## Step 5: Launch the Streamlit App

Run the following command from the project root directory:

```bash
streamlit run app.py
```

You should see output similar to:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

Your default web browser should open automatically. If it does not, open the **Local URL** shown in the terminal (`http://localhost:8501`).

---

## Step 6: Using the Dashboard

Once the app is running in your browser, you will see a sidebar on the left with four pages:

### Home
An overview of all three modules with descriptions of what each one does.

### Historical Analysis
1. Choose a **preset study** from the dropdown (e.g., "S&P 500 gains > 20%") or build a custom condition.
2. Adjust parameters in the sidebar (ticker, date range, thresholds).
3. Click **Run Study** to see results: forward return statistics, charts, and trigger dates.

### Watchlist Alerts
1. Choose **Default Watchlist** (19 major tickers) or enter a **Custom Watchlist**.
2. Click **Run Scan** to scan all tickers for technical signals.
3. Review the summary table with color-coded RSI, MACD, and Bollinger Band status.
4. Expand individual tickers to see detailed charts and alert breakdowns.

### ML Prediction
1. Enter a **ticker symbol** (e.g., AAPL, SPY).
2. Select the **forward horizon** (how many days ahead to predict).
3. Optionally enable the **LSTM model** (slower but provides an ensemble forecast).
4. Click **Run Prediction** to see the model output, feature importance, and consensus direction.

---

## Troubleshooting

### "streamlit: command not found"
Your virtual environment may not be active, or Streamlit did not install correctly.
- Re-activate the virtual environment (see Step 2).
- Reinstall: `pip install streamlit`
- Try running with: `python -m streamlit run app.py`

### "ModuleNotFoundError: No module named '...'"
A dependency is missing. Re-run:
```bash
pip install -r requirements.txt
```

### The app loads but charts are empty or show errors
This usually means data could not be fetched from Yahoo Finance. Check your internet connection and make sure the ticker symbols you entered are valid.

### Port 8501 is already in use
Another Streamlit instance may be running. Either stop it or specify a different port:
```bash
streamlit run app.py --server.port 8502
```

### PyTorch installation issues
If `torch` fails to install or is too large, install the CPU-only build:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Stopping the App

Press `Ctrl+C` in the terminal where Streamlit is running. This shuts down the local server. Close the browser tab at any time — it does not affect the server.

---

## Next Steps

- **Deploy publicly**: Push the project to GitHub and connect it to [Streamlit Community Cloud](https://share.streamlit.io) for free hosting.
- **CLI alternative**: Run `python main.py` for a terminal-based interface with the same modules.
- **Customize**: Edit `app.py` to add new pages, change layouts, or adjust default parameters.
