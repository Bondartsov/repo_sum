    import argparse, os, subprocess, time, psutil, json, pathlib, datetime, shlex, sys

    def parse_env_overrides(env_list):
        env = {}
        for item in env_list or []:
            if "=" not in item:
                raise ValueError(f"Bad --env item: {item}")
            k, v = item.split("=", 1)
            env[k.strip()] = v.strip()
        return env

    def run_with_metrics(index_cmd: str, env_overrides: dict, metrics_dir: pathlib.Path, profile: str, tag: str = None):
        metrics_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = tag or profile
        log_out = metrics_dir / f"{ts}_{tag}_stdout.log"
        log_err = metrics_dir / f"{ts}_{tag}_stderr.log"
        mem_csv = metrics_dir / f"{ts}_{tag}_mem.csv"
        summary_json = metrics_dir / f"{ts}_{tag}_summary.json"

        env = os.environ.copy()
        env.update(env_overrides)

        # Start process
        print(f"[run_ab] starting: {index_cmd}")
        with open(log_out, "w", encoding="utf-8") as fo, open(log_err, "w", encoding="utf-8") as fe:
            # On Windows use shell=True for complex commands; escape carefully
            shell = os.name == 'nt'
            start = time.time()
            proc = subprocess.Popen(index_cmd, shell=shell, env=env, stdout=fo, stderr=fe)

            # attach psutil to monitor child
            p = psutil.Process(proc.pid)
            # prime CPU percent
            try:
                p.cpu_percent(None)
            except Exception:
                pass

            # memory sampling loop
            with open(mem_csv, "w", encoding="utf-8") as fm:
                fm.write("t_seconds,rss_mb,vms_mb,cpu_percent
")
                peak_rss = 0.0
                while True:
                    if proc.poll() is not None:
                        # process ended; one last sample if possible
                        try:
                            rss = p.memory_info().rss / (1024*1024)
                            vms = p.memory_info().vms / (1024*1024)
                            cpu = p.cpu_percent(interval=0.0)
                            fm.write(f"{time.time()-start:.2f},{rss:.2f},{vms:.2f},{cpu:.2f}\n")
                            if rss > peak_rss: peak_rss = rss
                        except Exception:
                            pass
                        break
                    try:
                        rss = p.memory_info().rss / (1024*1024)
                        vms = p.memory_info().vms / (1024*1024)
                        cpu = p.cpu_percent(interval=0.5)
                        fm.write(f"{time.time()-start:.2f},{rss:.2f},{vms:.2f},{cpu:.2f}\n")
                        fm.flush()
                        if rss > peak_rss: peak_rss = rss
                    except psutil.NoSuchProcess:
                        break
                    except Exception:
                        time.sleep(0.5)
                retcode = proc.wait()
            elapsed = time.time() - start

        # System memory snapshot
        vm = psutil.virtual_memory()
        sys_mem = {
            "total_gb": round(vm.total / (1024**3), 2),
            "used_percent": vm.percent,
            "available_gb": round(vm.available / (1024**3), 2)
        }

        summary = {
            "profile": profile,
            "tag": tag,
            "index_cmd": index_cmd,
            "env_overrides": env_overrides,
            "started_at": ts,
            "elapsed_sec": round(elapsed, 2),
            "return_code": retcode,
            "peak_rss_mb": round(peak_rss, 2),
            "system_memory": sys_mem,
            "logs": {
                "stdout": str(log_out),
                "stderr": str(log_err),
                "mem_csv": str(mem_csv)
            }
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[run_ab] done: rc={retcode}, peak_rss_mb={peak_rss:.2f}, elapsed={elapsed:.2f}s")
        print(f"[run_ab] summary: {summary_json}")
        return retcode, summary_json

    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--index-cmd", required=True, help="Команда запуска индексации (например: "python .\\web_ui.py --index D:\\repo")")
        ap.add_argument("--metrics-dir", required=True, help="Папка для метрик (CSV/JSON/логи)")
        ap.add_argument("--profile", choices=["baseline","candidate"], required=True)
        ap.add_argument("--env", action="append", help="Переопределение переменных окружения, формат KEY=VALUE (можно много раз)")
        ap.add_argument("--tag", default=None, help="Доп. метка в имени файлов метрик")
        args = ap.parse_args()

        ret, summ = run_with_metrics(args.index_cmd, parse_env_overrides(args.env), pathlib.Path(args.metrics_dir), args.profile, args.tag)
        sys.exit(0 if ret == 0 else 1)

    if __name__ == "__main__":
        main()
