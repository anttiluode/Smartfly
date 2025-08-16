import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import queue
import time
from dataclasses import dataclass, field
import logging
import sys
import random

# -----------------------------
# The "Brain" - Oscillatory Cortex & CI Controller
# -----------------------------

@dataclass
class CortexParams:
    n_modules: int = 20
    n_workspace: int = 5
    ws_freq_hz: float = 6.0
    mod_base_hz: float = 24.0
    mod_sigma_hz: float = 1.0
    noise_level: float = 0.035
    mm_eps_hz: float = 0.04
    ws_eps_hz: float = 0.25
    pac_power: int = 2

@dataclass
class CIParams:
    w_ws: float = 0.4; w_plv: float = 0.4; w_pen: float = 0.2
    R_opt: float = 0.6
    CI_low: float = 0.55; CI_high: float = 0.80; CI_target: float = 0.675
    kmax_hz: float = 20.0; ki: float = 6.0; leak: float = 0.15
    focus_boost_hz: float = 8.0; focus_nudge: float = 0.12
    novelty_sigma: float = 0.7; novelty_frac: float = 0.5
    pulse_s: float = 0.25; cooldown_s: float = 0.6; hysteresis: float = 0.02

class CIController:
    def __init__(self, ci_params: CIParams):
        self.p = ci_params
        self.integral = 0.0
        self.pulse_end = -1e9
        self.last_event_time = -1e9
        self.mode = "IDLE"

    def compute_CI(self, R_ws, R, PLV):
        base = self.p.w_ws * R_ws + self.p.w_plv * PLV - self.p.w_pen * abs(R - self.p.R_opt)
        return float(np.clip(base, 0.0, 1.0))

    def decide(self, t, R_ws, R, PLV, dt):
        CI = self.compute_CI(R_ws, R, PLV)
        e = self.p.CI_target - CI
        self.integral += e * dt
        self.integral *= (1.0 - self.p.leak * dt)
        K_base = np.clip(self.p.ki * self.integral, 0.0, self.p.kmax_hz)
        event, K_eff, focus_nudge, nov_sigma = "NONE", K_base, 0.0, 0.0

        if t < self.pulse_end:
            if self.mode == "FOCUS":
                K_eff = np.clip(K_base + self.p.focus_boost_hz, 0.0, self.p.kmax_hz)
                focus_nudge = self.p.focus_nudge
            elif self.mode == "NOVELTY":
                K_eff = 0.6 * K_base
                nov_sigma = self.p.novelty_sigma
            event = f"PULSE_{self.mode}"
        elif (t - self.last_event_time) >= self.p.cooldown_s:
            if CI < (self.p.CI_low - self.p.hysteresis):
                self.mode = "FOCUS"; self.pulse_end = t + self.p.pulse_s; self.last_event_time = t; event = "FOCUS"
                K_eff = np.clip(K_base + self.p.focus_boost_hz, 0.0, self.p.kmax_hz)
                focus_nudge = self.p.focus_nudge
            elif CI > (self.p.CI_high + self.p.hysteresis):
                self.mode = "NOVELTY"; self.pulse_end = t + self.p.pulse_s; self.last_event_time = t; event = "NOVELTY"
                K_eff = 0.6 * K_base
                nov_sigma = self.p.novelty_sigma
            else:
                self.mode = "IDLE"
        return {'K_eff': K_eff, 'focus_nudge': focus_nudge, 'nov_sigma': nov_sigma, 'event': event, 'CI': CI}

class OscillatoryCortex:
    def __init__(self, p: CortexParams, ci: CIParams):
        self.p = p
        np.random.seed()
        self.ws_phase = np.random.uniform(0, 2 * np.pi, p.n_workspace)
        self.mod_phase = np.random.uniform(0, 2 * np.pi, p.n_modules)
        self.mod_freqs = p.mod_base_hz + p.mod_sigma_hz * np.random.randn(p.n_modules)
        self.time = 0.0
        self.controller = CIController(ci)
        self.R_ws, self.R_mod, self.PLV = 0.0, 0.0, 0.0

    def _wrap(self, x): return np.mod(x, 2 * np.pi)

    def process_vision(self, vision_cone):
        brightness = np.mean(vision_cone) / 255.0
        variance = np.var(vision_cone) / (255.0 * 128.0)
        ws_drive = brightness * 0.1
        mod_drive = variance * 0.2
        return ws_drive, mod_drive

    def update(self, vision_cone, dt):
        # 1. Process vision to get drives
        ws_drive, mod_drive = self.process_vision(vision_cone)
        
        # 2. Decide on control action based on *previous* state
        decision = self.controller.decide(self.time, self.R_ws, self.R_mod, self.PLV, dt)

        # 3. Evolve the oscillator states
        # Workspace update
        self.ws_phase += 2 * np.pi * self.p.ws_freq_hz * dt
        self.ws_phase += (ws_drive + self.p.noise_level) * np.random.randn(self.p.n_workspace) * np.sqrt(dt)
        mean_ws = np.angle(np.mean(np.exp(1j * self.ws_phase)))
        self.ws_phase += (2 * np.pi * self.p.ws_eps_hz) * np.sin(mean_ws - self.ws_phase) * dt
        if decision['focus_nudge'] > 0.0:
            self.ws_phase = (1.0 - decision['focus_nudge']) * self.ws_phase + decision['focus_nudge'] * mean_ws
        self.ws_phase = self._wrap(self.ws_phase)
        phi_ws = np.angle(np.mean(np.exp(1j * self.ws_phase)))
        
        # Module update
        if decision['nov_sigma'] > 0:
            idx = np.random.choice(self.p.n_modules, size=int(self.p.n_modules * self.controller.p.novelty_frac), replace=False)
            self.mod_phase[idx] += decision['nov_sigma'] * np.random.randn(len(idx))
            
        self.mod_phase += 2 * np.pi * self.mod_freqs * dt
        self.mod_phase += (mod_drive + self.p.noise_level) * np.random.randn(self.p.n_modules) * np.sqrt(dt)
        mean_mod_phase = np.angle(np.mean(np.exp(1j * self.mod_phase)))
        drive = 0.5 * (1.0 + np.cos(phi_ws)); drive = drive ** self.p.pac_power
        K_rad = 2 * np.pi * decision['K_eff']; MM_rad = 2 * np.pi * self.p.mm_eps_hz
        phase_diff = 4 * phi_ws - self.mod_phase; phase_diff = np.arctan2(np.sin(phase_diff), np.cos(phase_diff))
        self.mod_phase += K_rad * drive * np.sin(phase_diff) * dt
        self.mod_phase += MM_rad * np.sin(mean_mod_phase - self.mod_phase) * dt
        self.mod_phase = self._wrap(self.mod_phase)

        # 4. Update metrics for the next cycle
        self.R_ws = np.abs(np.mean(np.exp(1j * self.ws_phase)))
        self.R_mod = np.abs(np.mean(np.exp(1j * self.mod_phase)))
        self.PLV = np.abs(np.mean(np.exp(1j * (self.mod_phase - 4 * phi_ws))))
        
        self.time += dt
        return {'R_ws': self.R_ws, 'R_mod': self.R_mod, 'PLV': self.PLV, **decision}

# -----------------------------
# The "Body" - SmartFly Class
# -----------------------------

class SmartFly:
    def __init__(self, x, y, screen_width, screen_height):
        self.x, self.y = x, y
        self.screen_width, self.screen_height = screen_width, screen_height
        self.angle = random.uniform(0, 2 * np.pi)
        self.velocity = np.array([0.0, 0.0])
        self.size = 8.0
        self.momentum = 0.8
        
        # Vision parameters
        self.vision_cone_angle = np.pi / 2.5
        self.vision_cone_length = 200
        self.vision_res = (32, 32)
        
        # The Brain
        self.brain = OscillatoryCortex(CortexParams(), CIParams())
        
        # Behavior state
        self.action_timer = 0
        self.action_type = "NONE"

    def draw(self, frame):
        # Simple representation: a triangle for the body and a line for the head
        p1 = (int(self.x + self.size * np.cos(self.angle)), int(self.y + self.size * np.sin(self.angle)))
        p2 = (int(self.x + self.size * np.cos(self.angle + 2.3)), int(self.y + self.size * np.sin(self.angle + 2.3)))
        p3 = (int(self.x + self.size * np.cos(self.angle - 2.3)), int(self.y + self.size * np.sin(self.angle - 2.3)))
        
        # Color based on CI
        CI = self.brain.controller.compute_CI(self.brain.R_ws, self.brain.R_mod, self.brain.PLV)
        color_val = int(200 * CI + 55)
        color = (color_val, 255 - color_val, 150) # Blue when calm, green when chaotic
        
        cv2.line(frame, p2, p3, color, 1)
        cv2.line(frame, p1, p2, color, 1)
        cv2.line(frame, p1, p3, color, 1)
        
        # Head
        cv2.circle(frame, p1, int(self.size/3), (255, 100, 100), -1)

    def get_vision_cone(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        p1 = (int(self.x), int(self.y))
        p2 = (int(self.x + self.vision_cone_length * np.cos(self.angle - self.vision_cone_angle / 2)),
              int(self.y + self.vision_cone_length * np.sin(self.angle - self.vision_cone_angle / 2)))
        p3 = (int(self.x + self.vision_cone_length * np.cos(self.angle + self.vision_cone_angle / 2)),
              int(self.y + self.vision_cone_length * np.sin(self.angle + self.vision_cone_angle / 2)))
        
        cv2.fillPoly(mask, [np.array([p1, p2, p3])], 255)
        cone_image = cv2.bitwise_and(frame, frame, mask=mask)
        gray_cone = cv2.cvtColor(cone_image, cv2.COLOR_BGR2GRAY)
        
        return cv2.resize(gray_cone, self.vision_res)

    def update(self, frame, dt):
        vision_cone = self.get_vision_cone(frame)
        brain_state = self.brain.update(vision_cone, dt)
        
        # Interpret brain state to determine movement
        target_vel = np.array([0.0, 0.0])
        turn_speed = 0.0
        
        # Base speed is proportional to coupling strength (effort)
        base_speed = brain_state['K_eff'] * 0.5
        
        # Turn towards areas that increase workspace coherence
        # (A simple model: if R_ws is high, keep going straight, if low, turn)
        turn_impulse = (brain_state['R_ws'] - 0.5) * 2.0
        
        # Handle special actions triggered by controller events
        if self.action_timer > 0:
            self.action_timer -= dt
            if self.action_type == "DART":
                base_speed *= 4.0 # Dart forward
            elif self.action_type == "JERK":
                turn_impulse += (random.random() - 0.5) * 10.0 # Jerk randomly
        else: # If no action is active, check for new events
            if "FOCUS" in brain_state['event']:
                self.action_type = "DART"
                self.action_timer = 0.2
            elif "NOVELTY" in brain_state['event']:
                self.action_type = "JERK"
                self.action_timer = 0.15

        target_vel[0] = base_speed
        turn_speed = turn_impulse * 0.1
        
        # Apply rotation
        self.angle = self._wrap(self.angle + turn_speed * dt * 20.0, 0, 2 * np.pi)
        
        # Convert polar velocity to cartesian
        move_vec = np.array([np.cos(self.angle), np.sin(self.angle)]) * target_vel[0]
        
        # Apply momentum
        self.velocity = self.velocity * self.momentum + move_vec * (1 - self.momentum)
        
        self.x += self.velocity[0]
        self.y += self.velocity[1]
        
        # Wrap around screen edges
        self.x %= self.screen_width
        self.y %= self.screen_height

    def _wrap(self, val, min_val, max_val):
        return min_val + (val - min_val) % (max_val - min_val)

# -----------------------------
# The "World" - GUI and Camera
# -----------------------------

class SmartFlyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SmartFly Simulation")
        
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error", "Could not open camera.")
            sys.exit(1)
            
        self.width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.fly = SmartFly(self.width / 2, self.height / 2, self.width, self.height)
        
        self.setup_gui()
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self.last_time = time.time()
        
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
        
        self.process_frame()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left side for video
        self.canvas = tk.Canvas(main_frame, width=self.width, height=self.height, bg='black')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right side for controls and brain state
        control_panel = ttk.Frame(main_frame, padding="10")
        control_panel.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(control_panel, text="SmartFly Control Lab", font=("Helvetica", 16)).pack(pady=10)

        # Brain State Monitor
        brain_frame = ttk.LabelFrame(control_panel, text="Brain State [LIVE]", padding="10")
        brain_frame.pack(fill=tk.X, pady=10)
        
        self.brain_vars = {}
        metrics_to_show = ["CI", "K_eff", "R_ws", "R_mod", "PLV", "Event"]
        for metric in metrics_to_show:
            frame = ttk.Frame(brain_frame)
            frame.pack(fill=tk.X, pady=2)
            ttk.Label(frame, text=f"{metric}:", width=8, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value="--")
            ttk.Label(frame, textvariable=var, font=("Courier", 12)).pack(side=tk.LEFT)
            self.brain_vars[metric] = var

    def camera_loop(self):
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/60.0) # Cap camera capture rate

    def process_frame(self):
        try:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                
                current_time = time.time()
                dt = current_time - self.last_time
                self.last_time = current_time
                
                # Update the fly
                self.fly.update(frame, dt)
                
                # Draw the fly on the frame
                self.fly.draw(frame)
                
                # Update GUI
                self.update_display(frame)
                self.update_brain_monitor()

            self.root.after(16, self.process_frame) # Aim for ~60 FPS
        except Exception as e:
            logging.error(f"Processing error: {e}")
            self.cleanup()

    def update_display(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update_brain_monitor(self):
        state = self.fly.brain.controller.decide(self.fly.brain.time, self.fly.brain.R_ws, self.fly.brain.R_mod, self.fly.brain.PLV, 0.016)
        self.brain_vars["CI"].set(f"{state['CI']:.3f}")
        self.brain_vars["K_eff"].set(f"{state['K_eff']:.2f} Hz")
        self.brain_vars["R_ws"].set(f"{self.fly.brain.R_ws:.3f}")
        self.brain_vars["R_mod"].set(f"{self.fly.brain.R_mod:.3f}")
        self.brain_vars["PLV"].set(f"{self.fly.brain.PLV:.3f}")
        self.brain_vars["Event"].set(state['event'])

    def cleanup(self):
        self.running = False
        self.camera.release()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    root = tk.Tk()
    app = SmartFlyGUI(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()