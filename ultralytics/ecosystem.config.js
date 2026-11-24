module.exports = {
  apps: [
    {
      name: "monitor-server",

      cwd: "/home/iampong4/monitor_server/src",
      script: "index.js",

      interpreter: "node",
      instances: 1,
      exec_mode: "fork",

      autorestart: true,
      watch: false,
      max_memory_restart: "200M",
      restart_delay: 3000,

      env: {
        NODE_ENV: "production",
        PORT: 3000,
        MONITOR_API_KEY: process.env.MONITOR_API_KEY
      },

      time: true,
      out_file: "/home/iampong4/monitor_server/logs/out.log",
      error_file: "/home/iampong4/monitor_server/logs/error.log",
      merge_logs: true
    },
    {
      name: "ttsstt-server",

      cwd: "/home/iampong4/ttsstt_server",
      script: "src/server.js",

      interpreter: "node",
      instances: 1,
      exec_mode: "fork",

      autorestart: true,
      watch: false,
      max_memory_restart: "300M",
      restart_delay: 3000,
      // server stops its llama-server child on SIGINT (up to 15s), so give it time before SIGKILL
      kill_timeout: 20000,

      env: {
        NODE_ENV: "production",
        PORT: 3200,
        TTSSTT_AUTH_TOKEN: process.env.TTSSTT_AUTH_TOKEN,
        LLAMA_CPP_API_KEY: process.env.LLAMA_CPP_API_KEY
      },

      time: true,
      out_file: "/home/iampong4/ttsstt_server/logs/out.log",
      error_file: "/home/iampong4/ttsstt_server/logs/error.log",
      merge_logs: true
    },
    {
      // YOLO strip training. train_v2_pm2.py keeps a session (sessions/<name>/session.json + train.log):
      // it never starts twice, does nothing when the session is done, and resumes per fold after a crash or reboot.
      name: "strip-train",

      cwd: "/home/iampong4/yolo8_strip_25/ultralytics",
      script: "train_v2_pm2.py",
      // options of train_v2.py go here (command line wins over train_config.json), e.g. ["--only", "iris_*", "--epochs", "300"]
      args: [],

      interpreter: "/home/iampong4/venv-yolo/bin/python",
      interpreter_args: "-u",
      instances: 1,
      exec_mode: "fork",

      // restart after a crash/reboot; stop on exit 0 (all folds done, or session already done) and 3 (already running).
      // exit 1 = some fold failed: retry a few times only, otherwise a problem that cannot fix itself loops forever
      autorestart: true,
      stop_exit_codes: [0, 3],
      max_restarts: 5,
      min_uptime: 60000,
      restart_delay: 10000,
      watch: false,
      // SIGTERM is forwarded to the training child; give it time to stop before SIGKILL
      kill_timeout: 30000,

      time: true,
      out_file: "/home/iampong4/yolo8_strip_25/ultralytics/logs/pm2_out.log",
      error_file: "/home/iampong4/yolo8_strip_25/ultralytics/logs/pm2_error.log",
      merge_logs: true
    },
    {
      // Predictor that keeps the YOLO model loaded: drop a csv (or .jpg) into watch/inbox/ and read the answer
      // from watch/outbox/<name>.result.json. Start it after the model is trained (train_v2 / strip-train).
      name: "strip-predict",

      cwd: "/home/iampong4/yolo8_strip_25/ultralytics",
      script: "predict_watch_v2.py",
      args: ["--group", "iris_nopca_algo1_k8", "--fold", "0", "--device", "0"],

      interpreter: "/home/iampong4/venv-yolo/bin/python",
      interpreter_args: "-u",
      instances: 1,
      exec_mode: "fork",

      // exit 3 = another watcher already runs on this folder: do not restart
      autorestart: true,
      stop_exit_codes: [3],
      max_restarts: 10,
      min_uptime: 30000,
      restart_delay: 5000,
      watch: false,
      kill_timeout: 15000,

      time: true,
      out_file: "/home/iampong4/yolo8_strip_25/ultralytics/logs/predict_out.log",
      error_file: "/home/iampong4/yolo8_strip_25/ultralytics/logs/predict_error.log",
      merge_logs: true
    }
  ]
};
