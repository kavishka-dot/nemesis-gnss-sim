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

                # Default settings
                sim = NEMESISSimulator(
                    lat_deg=lat,
                    lon_deg=lon,
                    alt_m=10.0,
                    gps_tow=388800.0,
                    # We use embedded to ensure it works instantly,
                    # but could pass rinex if required.
                )

                truth = sim.compute_truth()

                # Generate 1ms of IQ data (so it returns fast to GUI)
                iq = sim.generate_iq(duration_ms=1.0)

                # Prepare JSON response
                satellites = []
                for sv in truth:
                    satellites.append({
                        "prn": sv.prn,
                        "el_deg": round(sv.el_deg, 2),
                        "az_deg": round(sv.az_deg, 2),
                        "doppler_hz": round(sv.doppler_hz, 2)
                    })

                # Downsample IQ for plotting if needed (4092 samples is fine for Chart.js though)
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
