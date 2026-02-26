#!/usr/bin/env node

import { execSync, spawnSync } from "child_process";
import os from "os";
import fs from "fs";
import path from "path";
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.join(__dirname, '..');
const mlModelDir = path.join(projectRoot, 'ml_model');
const uiDir = path.join(projectRoot, 'ui');
const venvDir = path.join(mlModelDir, 'venv');
const venvPython = os.platform() === 'win32'
    ? path.join(venvDir, 'Scripts', 'python.exe')
    : path.join(venvDir, 'bin', 'python');

console.log('\n========================================');
console.log('🧴 Skincare Recommendation System');
console.log('🚀 Automated Setup & Launch');
console.log('========================================\n');

// 1. Check Node.js
console.log('📋 Checking Node.js...');
try {
    const nodeVersion = execSync('node --version', { encoding: 'utf8' }).trim();
    console.log(`✓ Node.js is installed: ${nodeVersion}\n`);
} catch (err) {
    console.error('❌ Node.js is not installed. Please install Node.js 14+');
    console.error('   Download from: https://nodejs.org/');
    process.exit(1);
}

// 2. Check and install Node dependencies
console.log('📋 Checking Frontend dependencies...');
const packageLockExists = fs.existsSync(path.join(projectRoot, 'node_modules'));

if (!packageLockExists) {
    console.log('📥 Installing Frontend dependencies (npm install)...');
    try {
        execSync('npm install', {
            cwd: projectRoot,
            stdio: 'inherit'
        });
        console.log('✓ Frontend dependencies installed\n');
    } catch (err) {
        console.error('❌ Failed to install frontend dependencies');
        process.exit(1);
    }
} else {
    console.log('✓ Frontend dependencies already installed\n');
}

// 3. Check UI dependencies
console.log('📋 Checking UI (React) dependencies...');
const uiNodeModulesExists = fs.existsSync(path.join(uiDir, 'node_modules'));

if (!uiNodeModulesExists) {
    console.log('📥 Installing UI dependencies (npm install in ui/)...');
    try {
        execSync('npm install', {
            cwd: uiDir,
            stdio: 'inherit'
        });
        console.log('✓ UI dependencies installed\n');
    } catch (err) {
        console.error('❌ Failed to install UI dependencies');
        process.exit(1);
    }
} else {
    console.log('✓ UI dependencies already installed\n');
}

// 4. Check Python
console.log('📋 Checking Python...');
try {
    const pythonVersion = execSync('python --version', { encoding: 'utf8' }).trim();
    console.log(`✓ Python is installed: ${pythonVersion}\n`);
} catch (err) {
    console.error('❌ Python is not installed. Please install Python 3.8+');
    console.error('   Download from: https://www.python.org/downloads/');
    process.exit(1);
}

// 5. Check and create Python virtual environment
console.log('📋 Checking Python virtual environment...');
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
} else {
    console.log('✓ Virtual environment exists\n');
}

// 6. Check requirements.txt
console.log('📋 Checking Python dependencies...');
const requirementsFile = path.join(mlModelDir, 'requirements.txt');
if (!fs.existsSync(requirementsFile)) {
    console.error('❌ requirements.txt not found in ml_model directory');
    process.exit(1);
}

// 7. Check and install Python dependencies
try {
    const result = spawnSync(venvPython, ['-m', 'pip', 'list', '--format=json'], {
        cwd: mlModelDir,
        stdio: 'pipe',
        encoding: 'utf8'
    });

    if (result.status === 0) {
        try {
            const installed = JSON.parse(result.stdout);
            const installedNames = new Set(installed.map(p => p.name.toLowerCase()));

            // Check key dependencies
            const keyDeps = ['flask', 'pandas', 'numpy', 'scikit-learn', 'flask-cors'];
            const missingDeps = keyDeps.filter(dep => !installedNames.has(dep));

            if (missingDeps.length > 0) {
                console.log(`📥 Installing missing dependencies: ${missingDeps.join(', ')}`);
                execSync(`"${venvPython}" -m pip install -r requirements.txt`, {
                    cwd: mlModelDir,
                    stdio: 'inherit'
                });
                console.log('✓ Python dependencies installed\n');
            } else {
                console.log('✓ All Python dependencies are installed\n');
            }
        } catch (e) {
            throw new Error('Failed to parse pip list');
        }
    } else {
        throw new Error('Failed to list installed packages');
    }
} catch (err) {
    console.log('📥 Installing Python dependencies from requirements.txt...');
    try {
        execSync(`"${venvPython}" -m pip install -r requirements.txt`, {
            cwd: mlModelDir,
            stdio: 'inherit'
        });
        console.log('✓ Python dependencies installed\n');
    } catch (err) {
        console.error('❌ Failed to install Python dependencies');
        process.exit(1);
    }
}

// 8. Create .env file if it doesn't exist
console.log('📋 Checking environment configuration...');
const envFile = path.join(mlModelDir, '.env');
const envExampleFile = path.join(mlModelDir, '.env.example');
if (!fs.existsSync(envFile)) {
    if (fs.existsSync(envExampleFile)) {
        fs.copyFileSync(envExampleFile, envFile);
        console.log('✓ Created .env file from .env.example');
        console.log('⚠️  Please update .env with your Google OAuth credentials (optional for local testing)\n');
    }
} else {
    console.log('✓ .env configuration file exists\n');
}

// 9. All checks passed
console.log('========================================');
console.log('✓ All dependencies are installed!');
console.log('========================================\n');

console.log('📡 Starting Skincare Recommendation System...\n');
console.log('Backend  → http://localhost:5000');
console.log('Frontend → http://localhost:3000\n');

console.log('Press Ctrl+C to stop the servers\n');
console.log('========================================\n');

// Run concurrently
try {
    execSync('npm run dev', {
        cwd: projectRoot,
        stdio: 'inherit'
    });
} catch (err) {
    if (err.signal !== 'SIGINT') {
        console.error('❌ Error starting services:', err.message);
        process.exit(1);
    }
}
