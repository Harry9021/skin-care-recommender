import { execSync } from "child_process";
import os from "os";

try {
  const platform = os.platform();

  if (platform === "win32") {
    // PowerShell-compatible: no "call", just use full path
    execSync(`ml_model\\venv\\Scripts\\python.exe ml_model\\app.py`, {
      stdio: "inherit",
      shell: "powershell.exe"
    });
  } else {
    // macOS / Linux
    execSync(`source ml_model/venv/bin/activate && python ml_model/app.py`, {
      stdio: "inherit",
      shell: "/bin/bash"
    });
  }
} catch (err) {
  console.error("❌ Error starting ML model service:", err.message);
  process.exit(1);
}
