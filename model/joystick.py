import pygame
import numpy as np
import time
import threading
import collections

class JoystickController:
    def __init__(self, motion: str = "walk", deadzone: float = 0.1):
        """
        Initialize the joystick controller with a motion preset.
        
        Args:
            motion (str): "walk", "run", or "crawl". Determines speed limits.
            deadzone (float): Joystick deadzone threshold (0.0 ~ 1.0).
        """
        self.deadzone = deadzone
        self.motion_name = motion

        # command scale factors, defined by 99% percentile of the training data.
        velocity_range = {
            "walk":  {"vx": 1.661, "vy": 0.592, "wz": 2.3},
            "run":   {"vx": 2.639, "vy": 1.309, "wz": 2.008},
            # add more modes as needed
        }

        if motion in velocity_range:
            config = velocity_range[motion]
            self.max_vx = config["vx"]
            self.max_vy = config["vy"]
            self.max_wz = config["wz"]
        else:
            raise ValueError(f"Unknown motion mode: '{motion}'. Available: {list(velocity_range.keys())}")
        
        self.deadzone = deadzone
        self.joystick = None
        self._init_pygame()

    def _init_pygame(self):
        """Internal initialization of Pygame and the joystick."""
        if not pygame.get_init():
            pygame.init()
        if not pygame.joystick.get_init():
            pygame.joystick.init()

        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"✅ Joystick Connected: {self.joystick.get_name()}")
        else:
            print("❌ No joystick detected. Returning zero commands.")
            self.joystick = None

    def _apply_deadzone(self, value):
        """Apply deadzone logic to filter out noise."""
        if abs(value) < self.deadzone:
            return 0.0
        return value

    def get_command(self):
        """
        Read the current joystick state.
        
        Returns:
            np.array: [linear_x, linear_y, angular_z]
        """
        # Must call pump to process Pygame event queue
        pygame.event.pump()
        
        # Default command is zero
        cmd = np.zeros(3, dtype=np.float32)

        if self.joystick is not None:
            # Axis 0: Left Stick X (Horizontal)
            # Axis 1: Left Stick Y (Vertical)
            # Axis 2: Right Stick X (Horizontal)
            raw_lx = self.joystick.get_axis(0)
            raw_ly = self.joystick.get_axis(1)
            raw_rx = self.joystick.get_axis(2) # axis(3)

            # Prevent noise
            val_lx = self._apply_deadzone(raw_lx)
            val_ly = self._apply_deadzone(raw_ly)
            val_rx = self._apply_deadzone(raw_rx)

            # Scale to velocity ranges
            cmd[0] = -val_ly * self.max_vx # forward/backward
            cmd[1] = -val_rx * self.max_vy # left/right
            cmd[2] = -val_lx * self.max_wz # rotation

        vx, vy, wz = cmd
        status = "🛑 STOP" if np.all(cmd == 0) else "🟢 MOVE"
        
        # Print status in a single line
        print(f"\r{status} | "
                f"Vx: {vx:>+5.2f} | "
                f"Vy: {vy:>+5.2f} | "
                f"Wz: {wz:>+5.2f}    ", end="")
        
        return cmd

    def get_cond_commands(self, steps: int = 20, freq: float = 30.0) -> np.ndarray:
        """
        Capture a batch of commands at a fixed frequency.
        
        Args:
            steps (int): Number of command steps to record (default 20).
            freq (float): Sampling frequency in Hz (default 30Hz).
            
        Returns:
            np.ndarray: Array of shape (steps, 3) containing the commands.
        """
        command_history = []
        dt = 1.0 / freq  # Time per step
        
        for _ in range(steps):
            loop_start = time.perf_counter()
            
            cmd = self.get_command()
            command_history.append(cmd)
            vx, vy, wz = cmd
            status = "🛑 STOP" if np.all(cmd == 0) else "🟢 MOVE"
            
            # Print status in a single line
            print(f"\r{status} | "
                  f"Vx: {vx:>+5.2f} | "
                  f"Vy: {vy:>+5.2f} | "
                  f"Wz: {wz:>+5.2f}    ", end="")
            
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            #time.sleep(sleep_time) # uncomment for inference_rt.py
            
        # Stack into shape (steps, 3)
        return np.array(command_history, dtype=np.float32)
    
    def close(self):
        pygame.quit()