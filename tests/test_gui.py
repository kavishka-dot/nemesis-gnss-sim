import json
import threading
import socketserver
import urllib.request
import pytest
import time
from nemesis_sim.gui.server import NemesisGUIHandler

@pytest.fixture(scope="module")
def gui_server():
    # Find open port
    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True
        
    port = 8111
    while port < 8200:
        try:
            httpd = ReusableTCPServer(("127.0.0.1", port), NemesisGUIHandler)
            break
        except OSError:
            port += 1
            
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    # Wait for server to start
    time.sleep(0.5)
    
    yield f"http://127.0.0.1:{port}"
    
    httpd.shutdown()
    httpd.server_close()

def test_api_simulate_clearsky(gui_server):
    payload = {
        "lat": 6.9,
        "lon": 80.0,
        "attack_type": "none",
        "time_mode": "custom",
        "custom_time": "2025-01-01T12:00:00Z"
    }
    req = urllib.request.Request(f"{gui_server}/api/simulate", method="POST")
    req.add_header("Content-Type", "application/json")
    
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        
        assert "satellites" in data
        assert "iq_real" in data
        assert "iq_imag" in data
        assert "fs" in data
        assert len(data["iq_real"]) > 0
        assert len(data["iq_imag"]) > 0
        assert len(data["satellites"]) > 0

def test_api_simulate_adversarial(gui_server):
    payload = {
        "lat": 6.9,
        "lon": 80.0,
        "attack_type": "adversarial",
        "false_lat": 7.5,
        "false_lon": 81.0,
        "time_mode": "custom",
        "custom_time": "2025-01-01T12:00:00Z"
    }
    req = urllib.request.Request(f"{gui_server}/api/simulate", method="POST")
    req.add_header("Content-Type", "application/json")
    
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as response:
        assert response.status == 200
        data = json.loads(response.read().decode("utf-8"))
        
        assert "satellites" in data
        assert "iq_real" in data
        assert "iq_imag" in data

def test_gui_get_index(gui_server):
    req = urllib.request.Request(f"{gui_server}/", method="GET")
    with urllib.request.urlopen(req) as response:
        assert response.status == 200
        html = response.read().decode("utf-8")
        assert "<html" in html
        assert "NEMESIS" in html
