# What you find in this folder

- Tutorial to use virtual environments for Python in Windows

- requirements.txt file for venv virtual environments

- cellpose-env.yml file for Anaconda environments


# Venv and Python Script Setup Tutorial (for Windows Beginners)

## What you'll do:
1. Install Python
2. Download the project folder
3. Open Command Prompt as Administrator
4. Create a virtual environment
5. Install required libraries
6. Run the script

---

### Step 1: Install Python

1. Go to the official website: [https://www.python.org/downloads](https://www.python.org/downloads)
2. Download **Python 3.x (64-bit)** for Windows
3. **IMPORTANT:** During installation:
   - Check **"Add Python to PATH"**
   - At the end of the installation, click **"Disable path length limit"**  
     (to avoid problems saving files with long names on older versions of Windows)
4. Click **"Install Now"** and wait for it to finish.

---

### Step 2: Download the Project Folder

1. Go to this page:
 [https://github.com/lucaslovercio/ACHRI_Workshop_Cellpose](https://github.com/lucaslovercio/ACHRI_Workshop_Cellpose)
2. Click the green **"Code"** button, then click **"Download ZIP"**
3. After downloading, right-click the ZIP file and choose **"Extract All..."**
4. Open the extracted folder. You’ll use this folder in the next steps.

---

### Step 3: Open Command Prompt (as Administrator)

1. Click the **Start menu** (Windows icon in the bottom-left corner) 
2. Type **cmd** or **Command Prompt**
3. When "Command Prompt" appears, **right-click** on it
4. Select **"Run as administrator"**. A window may pop up asking for permission — click **"Yes"**.
5. Once the terminal is open, move into the project folder using the `cd` command. Example:

```cmd
cd C:\Users\YourName\Downloads\ACHRI_Workshop_Cellpose-main
```

Tip: You can open the folder in File Explorer, click in the address bar, copy the full path, and paste it after `cd`.

---

### Step 4: Create a Virtual Environment

Inside the project folder, run:

```cmd
python -m venv myenv
```

This will create a new folder called `myenv` containing your environment.

---

### Step 5: Activate the Environment

- **In Command Prompt**, activate it by running:

```cmd
myenv\Scripts\activate.bat
```

Once activated, you’ll see `(myenv)` at the beginning of the line.

---

### Step 6: Install Required Libraries

Make sure the file `requirements.txt` is in the folder. Then run:

```cmd
pip install -r requirements.txt
```

Wait a few moments — it will install everything needed for the script to work.

---

### Step 7: Run the Script

Now you can run the script:

```cmd
python your_script_name.py
```

Replace `your_script_name.py` with the actual name of the script you want them to run.

---

### Step 8: Exit the Environment

When you’re done, deactivate the environment by running:

```cmd
deactivate
```

---

### You're all set!
