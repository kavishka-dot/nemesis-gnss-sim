import http.server
import json
import os
import socketserver
import webbrowser

from nemesis_sim.simulator import NEMESISSimulator


class NemesisGUIHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve index.html from the gui directory when requesting /
        if self.path == "/":
            self.path = "/index.html"

        gui_dir = os.path.dirname(os.path.abspath(__file__))

        # Override the directory to serve files from the gui directory
        try:
            full_path = os.path.join(gui_dir, self.path.lstrip("/"))
            if os.path.isfile(full_path):
                with open(full_path, "rb") as f:
                    self.send_response(200)
                    if self.path.endswith(".html"):
                        self.send_header("Content-type", "text/html")
                    elif self.path.endswith(".css"):
                        self.send_header("Content-type", "text/css")
                    elif self.path.endswith(".js"):
                        self.send_header("Content-type", "application/javascript")
                    elif self.path.endswith(".png"):
                        self.send_header("Content-type", "image/png")
                    
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Expires", "0")
                    self.end_headers()
                    self.wfile.write(f.read())
                return
        except Exception:
            pass

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/simulate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            try:
                req = json.loads(post_data.decode("utf-8"))
                lat = float(req.get("lat", 0.0))
                lon = float(req.get("lon", 0.0))

                time_mode = req.get("time_mode", "realtime")
                custom_time = req.get("custom_time", "")
                
                from datetime import datetime, timezone
                if time_mode == "realtime" or not custom_time:
                    dt = datetime.now(timezone.utc)
                else:
                    try:
                        dt = datetime.fromisoformat(custom_time.replace('Z', '+00:00'))
                    except Exception:
                        dt = datetime.now(timezone.utc)

                doy = float(dt.timetuple().tm_yday)
                days_since_sunday = (dt.weekday() + 1) % 7
                gps_tow = days_since_sunday * 86400.0 + dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

                # Default settings
                sim = NEMESISSimulator(
                    lat_deg=lat,
                    lon_deg=lon,
                    alt_m=10.0,
                    gps_tow=gps_tow,
                    doy=doy
                )

                sim.compute_truth()

                from nemesis_sim.attacks import AttackConfig
                attack_type = req.get("attack_type", "none")
                if attack_type != "none":
                    if attack_type == "meaconing":
                        cfg = AttackConfig("meaconing", meaconing_delay_s=float(req.get("delay", 1e-4)))
                    elif attack_type == "slow_drift":
                        cfg = AttackConfig("slow_drift", drift_rate_m_s=float(req.get("drift_rate", 2.0)), drift_start_s=gps_tow)
                    elif attack_type == "adversarial":
                        cfg = AttackConfig("adversarial", false_lat=float(req.get("false_lat", lat)), false_lon=float(req.get("false_lon", lon)), false_alt=10.0)
                    else:
                        cfg = AttackConfig("none")
                    
                    if cfg.attack_type != "none":
                        sim.apply_attack(cfg)

                # Generate 1ms of IQ data
                iq = sim.generate_iq(duration_ms=1.0, use_attacked=(attack_type != "none"))

                # Prepare JSON response
                obs_list = sim._attack_obs if (attack_type != "none" and sim._attack_obs is not None) else sim._truth_obs
                satellites = []
                for sv in obs_list:
                    satellites.append({
                        "prn": sv.prn,
                        "el_deg": round(sv.el_deg, 2),
                        "az_deg": round(sv.az_deg, 2),
                        "doppler_hz": round(sv.doppler_hz, 2)
                    })

                # Downsample IQ for plotting
                iq_real = iq.real.tolist()
                iq_imag = iq.imag.tolist()

                response = {
                    "satellites": satellites,
                    "iq_real": iq_real,
                    "iq_imag": iq_imag,
                    "fs": sim.fs
                }

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode("utf-8"))

            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_gui(port=8080):
    # Enable address reuse and auto-find open port
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        
    while port < 8100:
        try:
            with ReusableTCPServer(("", port), NemesisGUIHandler) as httpd:
                print(f"Starting NEMESIS GUI on http://localhost:{port}")
                webbrowser.open(f"http://localhost:{port}")
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\nShutting down GUI server.")
                break
        except OSError as e:
            # If port is in use, try the next one
            if "10048" in str(e) or "10013" in str(e) or "Address already in use" in str(e):
                port += 1
            else:
                raise
