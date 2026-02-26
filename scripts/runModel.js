import { execSync, spawnSync, spawn } from "child_process";
import os from "os";
import fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, '..');
const mlModelDir = path.join(projectRoot, 'ml_model');
const venvDir = path.join(mlModelDir, 'venv');
const venvPython = os.platform() === 'win32'
  ? path.join(venvDir, 'Scripts', 'python.exe')
  : path.join(venvDir, 'bin', 'python');

console.log('🔍 Checking ML Model dependencies...\n');

// Check if Python is installed
try {
  execSync('python --version', { stdio: 'pipe' });
  console.log('✓ Python is installed');
} catch (err) {
  console.error('❌ Python is not installed. Please install Python 3.8+');
  process.exit(1);
}

// Check if virtual environment exists
if (!fs.existsSync(venvDir)) {
  console.log('📦 Creating Python virtual environment...');
  try {
    execSync(`python -m venv "${venvDir}"`, {
      cwd: mlModelDir,
      stdio: 'inherit'
    });
    console.log('✓ Virtual environment created\n');
  } catch (err) {
    console.error('❌ Failed to create virtual environment:', err.message);
    process.exit(1);
  }
}

// Check if requirements.txt exists
const requirementsFile = path.join(mlModelDir, 'requirements.txt');
if (!fs.existsSync(requirementsFile)) {
  console.error('❌ requirements.txt not found in ml_model directory');
  process.exit(1);
}

// Install/update Python dependencies
console.log('📦 Checking Python dependencies...');
try {
  const result = spawnSync(venvPython, ['-m', 'pip', 'list', '--format=json'], {
    cwd: mlModelDir,
    stdio: 'pipe',
    encoding: 'utf8'
  });

  if (result.status === 0) {
    const installed = JSON.parse(result.stdout);
    const installedNames = new Set(installed.map(p => p.name.toLowerCase()));

    // Check key dependencies
    const keyDeps = ['flask', 'pandas', 'numpy', 'scikit-learn'];
    const missingDeps = keyDeps.filter(dep => !installedNames.has(dep));

    if (missingDeps.length > 0) {
      console.log(`⚠️  Installing missing dependencies: ${missingDeps.join(', ')}`);
      execSync(`"${venvPython}" -m pip install -r requirements.txt`, {
        cwd: mlModelDir,
        stdio: 'inherit'
      });
      console.log('✓ Dependencies installed\n');
    } else {
      console.log('✓ All dependencies are installed\n');
    }
  }
} catch (err) {
  console.log('📥 Installing Python dependencies from requirements.txt...');
  execSync(`"${venvPython}" -m pip install -r requirements.txt`, {
    cwd: mlModelDir,
    stdio: 'inherit'
  });
  console.log('✓ Dependencies installed\n');
}

// Check if .env exists, if not create from .env.example
const envFile = path.join(mlModelDir, '.env');
const envExampleFile = path.join(mlModelDir, '.env.example');
if (!fs.existsSync(envFile)) {
  if (fs.existsSync(envExampleFile)) {
    fs.copyFileSync(envExampleFile, envFile);
    console.log('✓ Created .env file from .env.example');
  }
}

console.log('✓ ML Model setup complete!\n');

// Run the ML model
console.log('🚀 Starting ML Model Service...\n');

const appPath = path.join(mlModelDir, 'app.py');
const childProcess = spawn(venvPython, [appPath], {
  cwd: mlModelDir,
  stdio: 'inherit',
  shell: false
});

childProcess.on('error', (err) => {
  console.error('❌ Error starting ML model service:', err.message);
  process.exit(1);
});

childProcess.on('close', (code) => {
  if (code !== 0 && code !== null) {
    process.exit(code);
  }
});
