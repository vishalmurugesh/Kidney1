import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware for parsing JSON with safety limit
app.use(express.json({ limit: '500kb' }));

// Custom JSON body-parser error handler to prevent HTML stack leaks
app.use((err: any, _req: Request, res: Response, next: NextFunction) => {
  if (err instanceof SyntaxError && 'body' in err) {
    return res.status(400).json({
      success: false,
      error: 'Malformed JSON payload. Please ensure request body is valid JSON.'
    });
  }
  next();
});

// CORS headers configuration
app.use((req: Request, res: Response, next: NextFunction) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept');
  if (req.method === 'OPTIONS') {
    res.sendStatus(200);
    return;
  }
  next();
});

// Helper to run python prediction worker with safety timeout
function runPrediction(payload: Record<string, unknown>): Promise<any> {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, 'predict_cli.py');
    const pyProcess = spawn('python3', [pythonScript]);

    let stdoutData = '';
    let stderrData = '';
    let isCompleted = false;

    // Safety timeout: 8 seconds maximum
    const timer = setTimeout(() => {
      if (!isCompleted) {
        isCompleted = true;
        pyProcess.kill();
        reject(new Error('Prediction computation timed out. Please try again.'));
      }
    }, 8000);

    pyProcess.stdout.on('data', (data) => {
      stdoutData += data.toString();
    });

    pyProcess.stderr.on('data', (data) => {
      stderrData += data.toString();
    });

    pyProcess.on('error', (err) => {
      if (!isCompleted) {
        isCompleted = true;
        clearTimeout(timer);
        reject(new Error(`Failed to initiate prediction process: ${err.message}`));
      }
    });

    pyProcess.on('close', (code) => {
      if (isCompleted) return;
      isCompleted = true;
      clearTimeout(timer);

      if (code !== 0 && !stdoutData.trim()) {
        reject(new Error('Prediction engine exited with an unhandled status.'));
        return;
      }

      try {
        const parsed = JSON.parse(stdoutData.trim());
        resolve(parsed);
      } catch (err) {
        reject(new Error('Unexpected response format from prediction model service.'));
      }
    });

    // Write sanitized JSON payload to stdin
    pyProcess.stdin.write(JSON.stringify(payload));
    pyProcess.stdin.end();
  });
}

// Health check endpoint
app.get('/api/health', (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    project: 'Chronic Kidney Disease Prediction System',
    models: ['random_forest', 'adaboost', 'logistic_regression'],
    service: 'Active'
  });
});

// Models metadata endpoint
app.get('/api/models', (_req: Request, res: Response) => {
  res.json({
    models: [
      {
        id: 'random_forest',
        name: 'Random Forest Classifier',
        accuracy: '100.0%',
        recall: '100.0%',
        precision: '100.0%',
        f1Score: '100.0%',
        description: 'Ensemble of 100 decision trees evaluating non-linear orthogonal splits without feature correlation bias.'
      },
      {
        id: 'adaboost',
        name: 'AdaBoost Classifier',
        accuracy: '98.75%',
        recall: '98.0%',
        precision: '100.0%',
        f1Score: '98.99%',
        description: 'Adaptive boosting combining sequential decision stumps focused on hard-to-classify samples.'
      },
      {
        id: 'logistic_regression',
        name: 'Logistic Regression',
        accuracy: '98.75%',
        recall: '98.0%',
        precision: '100.0%',
        f1Score: '98.99%',
        description: 'Standardized L2 regularized linear model delivering smooth, calibrated prediction probabilities.'
      }
    ]
  });
});

// Primary Prediction Controller (both /predict and /api/predict)
const handlePredict = async (req: Request, res: Response) => {
  try {
    const inputData = req.body;
    if (!inputData || typeof inputData !== 'object' || Object.keys(inputData).length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No patient clinical data provided. Please submit required medical parameters.'
      });
    }

    const result = await runPrediction(inputData);
    if (!result.success) {
      return res.status(400).json(result);
    }

    return res.status(200).json(result);
  } catch (error: any) {
    return res.status(500).json({
      success: false,
      error: error?.message || 'An internal error occurred while processing the prediction request.'
    });
  }
};

app.post('/predict', handlePredict);
app.post('/api/predict', handlePredict);

// Dev / Prod Vite serving
async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.join(__dirname, 'dist')));
    app.get('*', (_req: Request, res: Response) => {
      res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
  }

  app.listen(PORT, () => {
    console.log(`Chronic Kidney Disease Prediction Server running on port ${PORT}`);
  });
}

startServer();
