"""AI Gesture Control System - Main Application.

Entry point with CustomTkinter GUI connecting video capture,
hand detection, gesture recognition, command execution,
performance monitoring, and user management.
"""

import cv2
import json
import logging
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, Optional

import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

# Project imports
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    APP_TITLE, APP_VERSION, AVAILABLE_COMMANDS, DEFAULT_GESTURE_MAPPINGS,
    Gestures, FRAME_WIDTH, FRAME_HEIGHT
)
from database.db_manager import DatabaseManager
from auth.auth_manager import AuthManager
from core.video_capture import VideoCapture
from core.hand_detector import HandDetector
from core.gesture_recognizer import GestureRecognizer
from core.command_executor import CommandExecutor
from training.gesture_trainer import GestureTrainer
from monitoring.performance_monitor import PerformanceMonitor

# ── Logging Setup ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gesture_ai.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ── CustomTkinter Theme ──
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class GestureAIApp(ctk.CTk):
    """Main application window with CustomTkinter GUI."""

    def __init__(self):
        super().__init__()
        
        # Window setup
        self.title(f"{APP_TITLE} v{APP_VERSION}")
        self.geometry("1200x800")
        self.minsize(1000, 700)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Core services
        self.db = DatabaseManager()
        self.auth = AuthManager(self.db)
        self.command_executor = CommandExecutor()
        
        # Pipeline components (initialized on start)
        self.video_capture: Optional[VideoCapture] = None
        self.hand_detector: Optional[HandDetector] = None
        self.gesture_recognizer: Optional[GestureRecognizer] = None
        self.gesture_trainer: Optional[GestureTrainer] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # State
        self._running = False
        self._camera_thread: Optional[threading.Thread] = None
        self._current_frame = None
        self._photo_image = None
        self._mouse_tracking = False
        self._training_mode = False
        self._training_gesture_name = ""
        self._training_samples_collected = 0
        self._training_target_samples = 100

        # Build UI
        self._build_login_screen()
        
        # Bring window to front/focus
        try:
            self.lift()
            self.attributes('-topmost', True)
            self.after(500, lambda: self.attributes('-topmost', False))
            self.focus_force()
        except Exception as e:
            logger.warning(f"Could not bring window to front: {e}")
        
        logger.info("GestureAIApp initialized")

    # ══════════════════════════════════════════════
    # LOGIN / REGISTER SCREEN
    # ══════════════════════════════════════════════
    
    def _build_login_screen(self):
        """Build the login/register screen."""
        # Clear window
        for widget in self.winfo_children():
            widget.destroy()
        
        self._login_frame = ctk.CTkFrame(self, corner_radius=15)
        self._login_frame.place(relx=0.5, rely=0.5, anchor="center")
        
        # Title
        title_label = ctk.CTkLabel(
            self._login_frame, text="\U0001F916 Gesture AI",
            font=ctk.CTkFont(size=32, weight="bold")
        )
        title_label.pack(pady=(30, 5))
        
        subtitle = ctk.CTkLabel(
            self._login_frame, text="Hand Gesture Control System",
            font=ctk.CTkFont(size=14), text_color="gray"
        )
        subtitle.pack(pady=(0, 20))
        
        # Tabview for Login/Register
        self._auth_tabs = ctk.CTkTabview(self._login_frame, width=350, height=280)
        self._auth_tabs.pack(padx=30, pady=10)
        
        login_tab = self._auth_tabs.add("Login")
        register_tab = self._auth_tabs.add("Register")
        
        # ── Login Tab ──
        self._login_username = ctk.CTkEntry(login_tab, placeholder_text="Username", width=280)
        self._login_username.pack(pady=(20, 10))
        
        self._login_password = ctk.CTkEntry(login_tab, placeholder_text="Password", show="•", width=280)
        self._login_password.pack(pady=(0, 15))
        
        login_btn = ctk.CTkButton(login_tab, text="Login", command=self._handle_login, width=280)
        login_btn.pack(pady=(0, 10))
        
        # ── Register Tab ──
        self._reg_username = ctk.CTkEntry(register_tab, placeholder_text="Username (3+ chars)", width=280)
        self._reg_username.pack(pady=(20, 10))
        
        self._reg_password = ctk.CTkEntry(register_tab, placeholder_text="Password (6+ chars)", show="•", width=280)
        self._reg_password.pack(pady=(0, 10))
        
        self._reg_password2 = ctk.CTkEntry(register_tab, placeholder_text="Confirm Password", show="•", width=280)
        self._reg_password2.pack(pady=(0, 15))
        
        reg_btn = ctk.CTkButton(register_tab, text="Register", command=self._handle_register, width=280)
        reg_btn.pack(pady=(0, 10))
        
        # Status message
        self._auth_status = ctk.CTkLabel(
            self._login_frame, text="", text_color="#FF6B6B",
            font=ctk.CTkFont(size=12), wraplength=300
        )
        self._auth_status.pack(pady=(5, 20))
        
        # Version info
        ver_label = ctk.CTkLabel(
            self._login_frame, text=f"v{APP_VERSION}",
            font=ctk.CTkFont(size=10), text_color="gray"
        )
        ver_label.pack(pady=(0, 15))

    def _handle_login(self):
        username = self._login_username.get()
        password = self._login_password.get()
        success, message = self.auth.login(username, password)
        
        if success:
            self._auth_status.configure(text=message, text_color="#4CAF50")
            self.after(500, self._build_main_dashboard)
        else:
            self._auth_status.configure(text=message, text_color="#FF6B6B")

    def _handle_register(self):
        username = self._reg_username.get()
        password = self._reg_password.get()
        password2 = self._reg_password2.get()
        
        if password != password2:
            self._auth_status.configure(text="Passwords do not match.", text_color="#FF6B6B")
            return
        
        success, message = self.auth.register(username, password)
        if success:
            self._auth_status.configure(text=message + " Please login.", text_color="#4CAF50")
            self._auth_tabs.set("Login")
        else:
            self._auth_status.configure(text=message, text_color="#FF6B6B")

    # ══════════════════════════════════════════════
    # MAIN DASHBOARD
    # ══════════════════════════════════════════════
    
    def _build_main_dashboard(self):
        """Build the main dashboard after login."""
        for widget in self.winfo_children():
            widget.destroy()
        
        user = self.auth.get_current_user()
        user_id = user['user_id']
        
        # Initialize pipeline components
        self.gesture_recognizer = GestureRecognizer(user_id=user_id)
        self.gesture_trainer = GestureTrainer(self.db, user_id)
        self.performance_monitor = PerformanceMonitor(self.db, user_id)
        
        # ── Top Bar ──
        top_bar = ctk.CTkFrame(self, height=50, corner_radius=0)
        top_bar.pack(fill="x", padx=0, pady=0)
        top_bar.pack_propagate(False)
        
        app_label = ctk.CTkLabel(
            top_bar, text=f"\U0001F916 Gesture AI",
            font=ctk.CTkFont(size=18, weight="bold")
        )
        app_label.pack(side="left", padx=15)
        
        user_label = ctk.CTkLabel(
            top_bar, text=f"\U0001F464 {user['username']}",
            font=ctk.CTkFont(size=13)
        )
        user_label.pack(side="right", padx=10)
        
        logout_btn = ctk.CTkButton(
            top_bar, text="Logout", width=80, height=30,
            fg_color="#555", hover_color="#777",
            command=self._handle_logout
        )
        logout_btn.pack(side="right", padx=5)
        
        # ── Main Content ──
        content = ctk.CTkFrame(self, corner_radius=0)
        content.pack(fill="both", expand=True, padx=10, pady=(5, 10))
        
        # Left: Camera Feed
        self._camera_frame = ctk.CTkFrame(content, corner_radius=10)
        self._camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self._camera_label = ctk.CTkLabel(self._camera_frame, text="Camera Feed")
        self._camera_label.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Camera controls
        cam_controls = ctk.CTkFrame(self._camera_frame, height=50)
        cam_controls.pack(fill="x", padx=10, pady=(0, 10))
        
        self._start_btn = ctk.CTkButton(
            cam_controls, text="▶ Start", width=100,
            fg_color="#2E7D32", hover_color="#388E3C",
            command=self._start_camera
        )
        self._start_btn.pack(side="left", padx=5)
        
        self._stop_btn = ctk.CTkButton(
            cam_controls, text="■ Stop", width=100,
            fg_color="#C62828", hover_color="#D32F2F",
            command=self._stop_camera, state="disabled"
        )
        self._stop_btn.pack(side="left", padx=5)
        
        self._mouse_btn = ctk.CTkButton(
            cam_controls, text="\U0001F5B1 Mouse: OFF", width=120,
            fg_color="#555", hover_color="#777",
            command=self._toggle_mouse_tracking
        )
        self._mouse_btn.pack(side="left", padx=5)
        
        self._status_label = ctk.CTkLabel(
            cam_controls, text="Ready",
            font=ctk.CTkFont(size=12), text_color="gray"
        )
        self._status_label.pack(side="right", padx=10)
        
        # Reaction speed (cooldown) control row
        speed_frame = ctk.CTkFrame(self._camera_frame)
        speed_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        ctk.CTkLabel(speed_frame, text="\u26A1 Reaction Speed:",
                     font=ctk.CTkFont(size=12)).pack(side="left", padx=5)
        
        self._cooldown_label = ctk.CTkLabel(
            speed_frame, text="300 ms",
            font=ctk.CTkFont(size=12, weight="bold"), width=60
        )
        self._cooldown_label.pack(side="right", padx=5)
        
        self._cooldown_slider = ctk.CTkSlider(
            speed_frame, from_=50, to=1500,
            number_of_steps=29, width=200,
            command=self._on_cooldown_changed
        )
        self._cooldown_slider.set(300)
        self._cooldown_slider.pack(side="right", padx=5)
        
        # Right: Sidebar with tabs
        sidebar = ctk.CTkFrame(content, width=350, corner_radius=10)
        sidebar.pack(side="right", fill="y", padx=(5, 0))
        sidebar.pack_propagate(False)
        
        self._sidebar_tabs = ctk.CTkTabview(sidebar, width=330)
        self._sidebar_tabs.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Tab 1: Gesture Mappings
        mappings_tab = self._sidebar_tabs.add("Mappings")
        self._build_mappings_tab(mappings_tab)
        
        # Tab 2: Training
        training_tab = self._sidebar_tabs.add("Training")
        self._build_training_tab(training_tab)
        
        # Tab 3: Custom Actions
        actions_tab = self._sidebar_tabs.add("Actions")
        self._build_actions_tab(actions_tab)
        
        # Tab 4: Statistics
        stats_tab = self._sidebar_tabs.add("Stats")
        self._build_stats_tab(stats_tab)
        
        # Load user's custom actions into CommandExecutor
        self._load_custom_actions()

    def _load_custom_actions(self):
        """Load user's custom actions from DB into CommandExecutor."""
        if not self.command_executor:
            return
        custom_actions = self.auth.get_custom_actions()
        for action in custom_actions:
            self.command_executor.register_custom_action(
                action['action_name'], action['action_type'],
                action['action_data'], action['description']
            )

    def _get_all_command_options(self):
        """Get all available commands including custom actions."""
        options = list(AVAILABLE_COMMANDS.keys())
        custom_actions = self.auth.get_custom_actions()
        for action in custom_actions:
            if action['action_name'] not in options:
                options.append(action['action_name'])
        return options

    def _build_mappings_tab(self, parent):
        """Build the gesture mappings configuration tab."""
        self._mappings_parent = parent
        for widget in parent.winfo_children():
            widget.destroy()

        header = ctk.CTkLabel(parent, text="Gesture \u2192 Command Mappings",
                              font=ctk.CTkFont(size=14, weight="bold"))
        header.pack(pady=(10, 5))

        # Add custom gesture input row
        add_frame = ctk.CTkFrame(parent)
        add_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        self._custom_gesture_entry = ctk.CTkEntry(
            add_frame, placeholder_text="Gesture name...", width=130
        )
        self._custom_gesture_entry.pack(side="left", padx=3, pady=5)
        
        add_btn = ctk.CTkButton(
            add_frame, text="+ Add", width=50,
            command=self._add_custom_gesture_to_mappings
        )
        add_btn.pack(side="left", padx=2, pady=5)
        
        reset_btn = ctk.CTkButton(
            add_frame, text="Reset", width=50,
            fg_color="#555", hover_color="#777",
            command=self._reset_default_mappings
        )
        reset_btn.pack(side="right", padx=2, pady=5)

        refresh_btn = ctk.CTkButton(
            add_frame, text="\U0001F504", width=30,
            fg_color="#555", hover_color="#777",
            command=lambda: self._build_mappings_tab(parent)
        )
        refresh_btn.pack(side="right", padx=2, pady=5)
        
        # Scrollable frame for mappings
        scroll = ctk.CTkScrollableFrame(parent, height=350)
        scroll.pack(fill="both", expand=True, padx=5, pady=5)
        
        mappings = self.auth.get_gesture_mappings()
        
        # Merge custom trained gestures from dataset DB if not present in mappings
        if self.gesture_trainer:
            custom_gestures = self.gesture_trainer.get_gesture_list()
            for g_name in custom_gestures.keys():
                if g_name not in mappings:
                    self.auth.save_gesture_mapping(g_name, "none")
                    mappings[g_name] = "none"

        command_options = self._get_all_command_options()
        self._mapping_vars = {}
        
        for gesture_name, current_cmd in mappings.items():
            row = ctk.CTkFrame(scroll)
            row.pack(fill="x", pady=2)
            
            label = ctk.CTkLabel(row, text=gesture_name, width=110, anchor="w",
                                 font=ctk.CTkFont(size=12))
            label.pack(side="left", padx=5)
            
            # Ensure current_cmd is in the dropdown options
            if current_cmd and current_cmd not in command_options:
                command_options.append(current_cmd)
            
            var = ctk.StringVar(value=current_cmd)
            dropdown = ctk.CTkOptionMenu(row, values=command_options, variable=var, width=130)
            dropdown.pack(side="right", padx=5)
            self._mapping_vars[gesture_name] = var

            # Add delete button for EVERY gesture (built-in or custom)
            del_btn = ctk.CTkButton(
                row, text="\u2716", width=25, height=25,
                fg_color="#C62828", hover_color="#D32F2F",
                command=lambda g=gesture_name: self._remove_gesture(g)
            )
            del_btn.pack(side="right", padx=2)
        
        save_btn = ctk.CTkButton(parent, text="Save Mappings",
                                  command=self._save_mappings)
        save_btn.pack(pady=10)
        
        self._mapping_status = ctk.CTkLabel(parent, text="", font=ctk.CTkFont(size=11))
        self._mapping_status.pack()

    def _remove_gesture(self, gesture_name: str):
        """Remove any gesture completely from mappings, dataset, and trainer."""
        self.auth.delete_gesture_mapping(gesture_name)
        if self.gesture_trainer:
            self.gesture_trainer.delete_gesture(gesture_name)
            counts = self.gesture_trainer.get_gesture_list()
            if counts:
                self.gesture_trainer.train_model()
            else:
                self.gesture_trainer.delete_model()
        if self.gesture_recognizer:
            self.gesture_recognizer.reload_model()
        self._refresh_gesture_list()
        if hasattr(self, '_mappings_parent'):
            self._build_mappings_tab(self._mappings_parent)

    def _reset_default_mappings(self):
        """Reset mappings back to system defaults."""
        self.auth.reset_default_mappings()
        if hasattr(self, '_mappings_parent'):
            self._build_mappings_tab(self._mappings_parent)

    def _add_custom_gesture_to_mappings(self):
        """Add a custom gesture name to mappings manually."""
        name = self._custom_gesture_entry.get().strip()
        if name:
            self.auth.save_gesture_mapping(name, "none")
            self._build_mappings_tab(self._mappings_parent)

    def _save_mappings(self):
        """Save all gesture mappings."""
        for gesture_name, var in self._mapping_vars.items():
            self.auth.save_gesture_mapping(gesture_name, var.get())
        self._mapping_status.configure(text="\u2713 Mappings saved!", text_color="#4CAF50")
        self.after(3000, lambda: self._mapping_status.configure(text=""))

    # ══════════════════════════════════════════════
    # CUSTOM ACTIONS TAB
    # ══════════════════════════════════════════════

    def _build_actions_tab(self, parent):
        """Build the custom actions creation tab."""
        self._actions_parent = parent
        for widget in parent.winfo_children():
            widget.destroy()

        header = ctk.CTkLabel(parent, text="Custom Actions",
                              font=ctk.CTkFont(size=14, weight="bold"))
        header.pack(pady=(10, 5))

        ctk.CTkLabel(parent, text="Create actions you can assign to any gesture",
                     font=ctk.CTkFont(size=11), text_color="gray").pack(pady=(0, 5))

        # ── New Action Form ──
        form = ctk.CTkFrame(parent)
        form.pack(fill="x", padx=5, pady=5)

        # Action name
        name_row = ctk.CTkFrame(form)
        name_row.pack(fill="x", padx=5, pady=3)
        ctk.CTkLabel(name_row, text="Name:", width=60, anchor="w").pack(side="left")
        self._action_name_entry = ctk.CTkEntry(
            name_row, placeholder_text="e.g. open_discord", width=180
        )
        self._action_name_entry.pack(side="right", padx=5)

        # Action type
        type_row = ctk.CTkFrame(form)
        type_row.pack(fill="x", padx=5, pady=3)
        ctk.CTkLabel(type_row, text="Type:", width=60, anchor="w").pack(side="left")
        self._action_type_var = ctk.StringVar(value="hotkey")
        type_dropdown = ctk.CTkOptionMenu(
            type_row,
            values=["hotkey", "run_program", "open_url", "type_text", "press_key"],
            variable=self._action_type_var, width=180
        )
        type_dropdown.pack(side="right", padx=5)

        # Action data
        data_row = ctk.CTkFrame(form)
        data_row.pack(fill="x", padx=5, pady=3)
        ctk.CTkLabel(data_row, text="Value:", width=60, anchor="w").pack(side="left")
        self._action_data_entry = ctk.CTkEntry(
            data_row, placeholder_text="e.g. ctrl+shift+t", width=180
        )
        self._action_data_entry.pack(side="right", padx=5)

        # Description
        desc_row = ctk.CTkFrame(form)
        desc_row.pack(fill="x", padx=5, pady=3)
        ctk.CTkLabel(desc_row, text="Desc:", width=60, anchor="w").pack(side="left")
        self._action_desc_entry = ctk.CTkEntry(
            desc_row, placeholder_text="Open Discord app", width=180
        )
        self._action_desc_entry.pack(side="right", padx=5)

        # Create button
        create_btn = ctk.CTkButton(
            parent, text="+ Create Action",
            fg_color="#1565C0", hover_color="#1976D2",
            command=self._create_custom_action
        )
        create_btn.pack(pady=8)

        self._action_status = ctk.CTkLabel(parent, text="", font=ctk.CTkFont(size=11),
                                            wraplength=280)
        self._action_status.pack(pady=2)

        # ── Help Text ──
        help_frame = ctk.CTkFrame(parent)
        help_frame.pack(fill="x", padx=5, pady=5)
        help_text = (
            "Types & Examples:\n"
            "• hotkey: ctrl+shift+t, alt+f4, win+d\n"
            "• run_program: notepad.exe, discord.exe\n"
            "• open_url: https://youtube.com\n"
            "• type_text: Hello World\n"
            "• press_key: enter, tab, space, escape"
        )
        ctk.CTkLabel(help_frame, text=help_text, font=ctk.CTkFont(size=10),
                     justify="left", text_color="#aaa").pack(padx=8, pady=5)

        # ── Existing Actions List ──
        ctk.CTkLabel(parent, text="Your Custom Actions:",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(8, 3))
        
        actions_scroll = ctk.CTkScrollableFrame(parent, height=100)
        actions_scroll.pack(fill="x", padx=5, pady=5)

        custom_actions = self.auth.get_custom_actions()
        if not custom_actions:
            ctk.CTkLabel(actions_scroll, text="No custom actions yet",
                         text_color="gray").pack()
        else:
            for action in custom_actions:
                row = ctk.CTkFrame(actions_scroll)
                row.pack(fill="x", pady=1)
                
                info = f"{action['action_name']}  [{action['action_type']}]"
                ctk.CTkLabel(row, text=info, font=ctk.CTkFont(size=11),
                             anchor="w").pack(side="left", padx=5)
                
                del_btn = ctk.CTkButton(
                    row, text="\u2716", width=25, height=22,
                    fg_color="#C62828", hover_color="#D32F2F",
                    command=lambda n=action['action_name']: self._delete_custom_action(n)
                )
                del_btn.pack(side="right", padx=3)

    def _create_custom_action(self):
        """Create a new custom action from the form inputs."""
        name = self._action_name_entry.get().strip()
        action_type = self._action_type_var.get()
        data = self._action_data_entry.get().strip()
        desc = self._action_desc_entry.get().strip()

        if not name:
            self._action_status.configure(text="Enter an action name!", text_color="#FF6B6B")
            return
        if not data:
            self._action_status.configure(text="Enter a value!", text_color="#FF6B6B")
            return

        # Save to DB
        success, msg = self.auth.save_custom_action(name, action_type, data, desc)
        if success:
            # Register in CommandExecutor immediately
            if self.command_executor:
                self.command_executor.register_custom_action(name, action_type, data, desc)
            
            self._action_status.configure(
                text=f"\u2713 '{name}' created! Go to Mappings tab to assign it.",
                text_color="#4CAF50"
            )
            # Clear form
            self._action_name_entry.delete(0, 'end')
            self._action_data_entry.delete(0, 'end')
            self._action_desc_entry.delete(0, 'end')
            # Refresh actions list and mappings dropdown
            self._build_actions_tab(self._actions_parent)
            if hasattr(self, '_mappings_parent'):
                self._build_mappings_tab(self._mappings_parent)
        else:
            self._action_status.configure(text=f"\u2717 {msg}", text_color="#FF6B6B")

    def _delete_custom_action(self, action_name: str):
        """Delete a custom action from DB and CommandExecutor."""
        self.auth.delete_custom_action(action_name)
        if self.command_executor:
            self.command_executor.unregister_custom_action(action_name)
        self._build_actions_tab(self._actions_parent)
        if hasattr(self, '_mappings_parent'):
            self._build_mappings_tab(self._mappings_parent)

    def _build_training_tab(self, parent):
        """Build the custom gesture training tab."""
        header = ctk.CTkLabel(parent, text="Custom Gesture Training",
                              font=ctk.CTkFont(size=14, weight="bold"))
        header.pack(pady=(10, 5))
        
        # Gesture name input
        name_frame = ctk.CTkFrame(parent)
        name_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(name_frame, text="Gesture Name:").pack(side="left", padx=5)
        self._train_name_entry = ctk.CTkEntry(name_frame, width=150,
                                               placeholder_text="e.g., ThumbsDown")
        self._train_name_entry.pack(side="right", padx=5)
        
        # Samples count
        samples_frame = ctk.CTkFrame(parent)
        samples_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(samples_frame, text="Samples:").pack(side="left", padx=5)
        self._train_samples_val_label = ctk.CTkLabel(
            samples_frame, text="100", font=ctk.CTkFont(size=12, weight="bold"), width=35
        )
        self._train_samples_val_label.pack(side="right", padx=2)
        
        self._train_samples_slider = ctk.CTkSlider(
            samples_frame, from_=30, to=200,
            number_of_steps=17, width=120,
            command=self._on_samples_slider_changed
        )
        self._train_samples_slider.set(100)
        self._train_samples_slider.pack(side="right", padx=2)
        
        # Record button
        self._record_btn = ctk.CTkButton(
            parent, text="\U0001F534 Record Samples",
            fg_color="#C62828", hover_color="#D32F2F",
            command=self._start_recording
        )
        self._record_btn.pack(pady=10)
        
        # Progress bar
        self._train_progress = ctk.CTkProgressBar(parent, width=280)
        self._train_progress.pack(pady=5)
        self._train_progress.set(0)
        
        self._train_status = ctk.CTkLabel(parent, text="", font=ctk.CTkFont(size=11),
                                           wraplength=280)
        self._train_status.pack(pady=5)
        
        # Train model button
        train_btn = ctk.CTkButton(
            parent, text="\U0001F9E0 Train Model",
            fg_color="#1565C0", hover_color="#1976D2",
            command=self._train_model
        )
        train_btn.pack(pady=5)
        
        # Existing gestures list
        ctk.CTkLabel(parent, text="Trained Gestures:",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(10, 5))
        
        self._gestures_list_frame = ctk.CTkScrollableFrame(parent, height=120)
        self._gestures_list_frame.pack(fill="x", padx=10, pady=5)
        
        self._refresh_gesture_list()

    def _on_samples_slider_changed(self, value):
        """Update the sample count display label live when slider moves."""
        val = int(value)
        if hasattr(self, '_train_samples_val_label'):
            self._train_samples_val_label.configure(text=str(val))

    def _refresh_gesture_list(self):
        """Refresh the list of trained gestures."""
        for widget in self._gestures_list_frame.winfo_children():
            widget.destroy()
        
        if self.gesture_trainer:
            gestures = self.gesture_trainer.get_gesture_list()
            if not gestures:
                ctk.CTkLabel(self._gestures_list_frame, text="No custom gestures yet",
                             text_color="gray").pack()
            else:
                for name, count in gestures.items():
                    row = ctk.CTkFrame(self._gestures_list_frame)
                    row.pack(fill="x", pady=1)
                    ctk.CTkLabel(row, text=f"{name} ({count} samples)",
                                 font=ctk.CTkFont(size=11)).pack(side="left", padx=5)
                    del_btn = ctk.CTkButton(
                        row, text="\u2716", width=30, height=25,
                        fg_color="#C62828", hover_color="#D32F2F",
                        command=lambda n=name: self._delete_gesture(n)
                    )
                    del_btn.pack(side="right", padx=5)

    def _delete_gesture(self, name: str):
        self._remove_gesture(name)

    def _start_recording(self):
        """Start recording training samples."""
        name = self._train_name_entry.get().strip()
        if not name:
            self._train_status.configure(text="Enter a gesture name first!", text_color="#FF6B6B")
            return
        
        if not self._running:
            self._train_status.configure(text="Start the camera first!", text_color="#FF6B6B")
            return
        
        self._training_mode = True
        self._training_gesture_name = name
        self.auth.save_gesture_mapping(name, "none")
        if hasattr(self, '_mappings_parent'):
            self._build_mappings_tab(self._mappings_parent)
        self._training_samples_collected = 0
        self._training_target_samples = int(self._train_samples_slider.get())
        self._train_progress.set(0)
        self._record_btn.configure(state="disabled", text="Recording...")
        self._train_status.configure(
            text=f"Hold '{name}' gesture... (0/{self._training_target_samples})",
            text_color="#FFA726"
        )

    def _train_model(self):
        """Train ML model on collected samples."""
        if not self.gesture_trainer:
            return
        
        self._train_status.configure(text="Training model...", text_color="#FFA726")
        self.update_idletasks()
        
        # Run training in background thread
        def do_train():
            result = self.gesture_trainer.train_model()
            self.after(0, lambda: self._on_training_complete(result))
        
        threading.Thread(target=do_train, daemon=True).start()

    def _on_training_complete(self, result: Dict):
        """Handle training completion."""
        if result['success']:
            self._train_status.configure(
                text=f"\u2713 {result['message']}\n"
                     f"Accuracy: {result['accuracy']:.1%}\n"
                     f"Model: {result['best_model']}",
                text_color="#4CAF50"
            )
            # Reload model in recognizer
            if self.gesture_recognizer:
                self.gesture_recognizer.reload_model()
        else:
            self._train_status.configure(
                text=f"\u2717 {result['message']}",
                text_color="#FF6B6B"
            )
        self._refresh_gesture_list()
        if hasattr(self, '_mappings_parent'):
            self._build_mappings_tab(self._mappings_parent)

    def _build_stats_tab(self, parent):
        """Build the session statistics tab."""
        header = ctk.CTkLabel(parent, text="Session Statistics",
                              font=ctk.CTkFont(size=14, weight="bold"))
        header.pack(pady=(10, 10))
        
        self._stats_text = ctk.CTkTextbox(parent, height=350, width=300,
                                           font=ctk.CTkFont(family="Consolas", size=12))
        self._stats_text.pack(padx=10, pady=5, fill="both", expand=True)
        self._stats_text.configure(state="disabled")
        
        refresh_btn = ctk.CTkButton(parent, text="\U0001F504 Refresh Stats",
                                     command=self._refresh_stats)
        refresh_btn.pack(pady=5)
        
        export_btn = ctk.CTkButton(parent, text="\U0001F4BE Export Log",
                                    fg_color="#555", hover_color="#777",
                                    command=self._export_log)
        export_btn.pack(pady=(0, 10))

    def _refresh_stats(self):
        """Refresh session statistics display."""
        if not self.performance_monitor:
            return
        
        stats = self.performance_monitor.get_session_stats()
        
        text = f"""\u250c\u2500\u2500 Session Overview \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
\u2502 Duration:     {stats['session_duration_str']}        \u2502
\u2502 Total Gestures: {stats['total_gestures']:<8d}     \u2502
\u2502 Avg FPS:        {stats['avg_fps']:<8.1f}     \u2502
\u2502 Avg Confidence: {stats['avg_confidence']:<8.1%}   \u2502
\u2502 Most Common:    {stats['most_common_gesture']:<14s} \u2502
\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518

\u250c\u2500\u2500 Gesture Breakdown \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510
"""
        for gesture, count in sorted(stats['gesture_counts'].items(), key=lambda x: -x[1]):
            bar_len = min(15, int(count / max(stats['total_gestures'], 1) * 15))
            bar = '\u2588' * bar_len + '\u2591' * (15 - bar_len)
            text += f"\u2502 {gesture:<14s} {bar} {count:>4d} \u2502\n"
        
        text += "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518"
        
        self._stats_text.configure(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("1.0", text)
        self._stats_text.configure(state="disabled")

    def _export_log(self):
        if self.performance_monitor:
            path = self.performance_monitor.export_session_log()
            if path:
                self._train_status.configure(text=f"Log exported: {path.name}", text_color="#4CAF50")

    # ══════════════════════════════════════════════
    # CAMERA & GESTURE PIPELINE
    # ══════════════════════════════════════════════
    
    def _start_camera(self):
        """Start the camera and gesture processing pipeline."""
        if self._running:
            return
        
        self.video_capture = VideoCapture()
        if not self.video_capture.open():
            self._status_label.configure(text="\u2717 Camera failed!", text_color="#FF6B6B")
            return
        
        self.hand_detector = HandDetector()
        if self.performance_monitor:
            self.performance_monitor.reset()
        
        self._running = True
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._status_label.configure(text="\u25CF Running", text_color="#4CAF50")
        
        # Start processing in background thread
        self._camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._camera_thread.start()
        
        # Start UI update loop
        self._update_ui()

    def _stop_camera(self):
        """Stop the camera and gesture processing."""
        if not self._running and self.video_capture is None:
            return  # Already stopped
        
        self._running = False
        self._training_mode = False
        
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
            self._camera_thread = None
        
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        if self.hand_detector:
            self.hand_detector.close()
            self.hand_detector = None
        if self.performance_monitor:
            self.performance_monitor.flush()
        
        # Update UI elements (may not exist during shutdown)
        try:
            self._start_btn.configure(state="normal")
            self._stop_btn.configure(state="disabled")
            self._status_label.configure(text="Stopped", text_color="gray")
            self._camera_label.configure(image=None, text="Camera Stopped")
        except (AttributeError, Exception):
            pass  # UI widgets may not exist during app shutdown

    def _camera_loop(self):
        """Background thread: capture frames and run gesture pipeline."""
        while self._running:
            if not self.video_capture or not self.video_capture.is_opened():
                break
            
            success, frame = self.video_capture.read_frame()
            if not success:
                continue
            
            # Fetch active mappings dynamically on each frame
            gesture_mappings = self.auth.get_gesture_mappings()
            
            # Convert to RGB for MediaPipe
            rgb_frame = VideoCapture.to_rgb(frame)
            
            # Detect hands
            hands = []
            if self.hand_detector:
                hands = self.hand_detector.detect(rgb_frame)
            
            gesture_result = None
            
            for hand in hands:
                # Draw landmarks
                HandDetector.draw_landmarks(frame, hand)
                
                # Training mode: record samples
                if self._training_mode and self.gesture_trainer:
                    if self._training_samples_collected < self._training_target_samples:
                        if self.gesture_trainer.record_sample(
                            self._training_gesture_name, hand
                        ):
                            self._training_samples_collected += 1
                    else:
                        self._training_mode = False
                        self.after(0, self._on_recording_complete)
                    continue
                
                # Gesture recognition — always detect, but filter when paused or unmapped
                if self.gesture_recognizer:
                    raw_result = self.gesture_recognizer.recognize(hand)
                    
                    if raw_result:
                        command_key = gesture_mappings.get(raw_result.gesture_name)
                        
                        # Only execute and display if the gesture has an ACTIVE, non-none mapping!
                        if command_key and command_key != 'none':
                            # When paused, only allow pause_resume to go through
                            if self.command_executor.is_paused and command_key != 'pause_resume':
                                continue
                            
                            gesture_result = raw_result
                            context = {
                                'hand': hand,
                                'frame_width': frame.shape[1],
                                'frame_height': frame.shape[0]
                            }
                            
                            # Mouse move is continuous (no cooldown needed)
                            if command_key == 'mouse_move' and self._mouse_tracking:
                                self.command_executor.execute(command_key, context)
                            elif command_key != 'mouse_move':
                                success_exec, msg = self.command_executor.execute(command_key, context)
                                if success_exec and self.performance_monitor:
                                    self.performance_monitor.log_event(
                                        gesture_result.gesture_name,
                                        command_key,
                                        gesture_result.confidence,
                                        gesture_result.hand_label
                                    )
            
            # Update FPS
            if self.performance_monitor:
                fps = self.performance_monitor.update_fps()
                
                # Draw OSD overlay (only for active, mapped gestures!)
                g_name = gesture_result.gesture_name if gesture_result else None
                g_conf = gesture_result.confidence if gesture_result else None
                self.performance_monitor.draw_overlay(
                    frame, g_name, g_conf, self.command_executor.is_paused
                )
            
            # Store frame for UI update
            self._current_frame = frame
            
            # Small sleep to prevent CPU overuse
            time.sleep(0.001)

    def _on_recording_complete(self):
        """Called when training sample recording is finished."""
        self._record_btn.configure(state="normal", text="\U0001F534 Record Samples")
        self._train_status.configure(
            text=f"\u2713 Recorded {self._training_samples_collected} samples for "
                 f"'{self._training_gesture_name}'",
            text_color="#4CAF50"
        )
        self._train_progress.set(1.0)
        self._refresh_gesture_list()

    def _update_ui(self):
        """Update the camera feed display in the GUI (runs on main thread)."""
        if not self._running:
            return
        
        frame = self._current_frame
        if frame is not None:
            # Convert BGR to RGB for PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to fit the label
            label_w = self._camera_label.winfo_width()
            label_h = self._camera_label.winfo_height()
            if label_w > 10 and label_h > 10:
                # Maintain aspect ratio
                frame_h, frame_w = rgb.shape[:2]
                scale = min(label_w / frame_w, label_h / frame_h)
                new_w = int(frame_w * scale)
                new_h = int(frame_h * scale)
                rgb = cv2.resize(rgb, (new_w, new_h))
            
            img = Image.fromarray(rgb)
            self._photo_image = ImageTk.PhotoImage(img)
            self._camera_label.configure(image=self._photo_image, text="")
        
        # Update training progress
        if self._training_mode and self._training_target_samples > 0:
            progress = self._training_samples_collected / self._training_target_samples
            self._train_progress.set(progress)
            self._train_status.configure(
                text=f"Recording '{self._training_gesture_name}'... "
                     f"({self._training_samples_collected}/{self._training_target_samples})"
            )
        
        # Schedule next update (~30 FPS for UI)
        self.after(33, self._update_ui)

    def _toggle_mouse_tracking(self):
        """Toggle mouse tracking on/off."""
        self._mouse_tracking = not self._mouse_tracking
        self.command_executor.set_mouse_tracking(self._mouse_tracking)
        if self._mouse_tracking:
            self._mouse_btn.configure(
                text="\U0001F5B1 Mouse: ON",
                fg_color="#2E7D32", hover_color="#388E3C"
            )
        else:
            self._mouse_btn.configure(
                text="\U0001F5B1 Mouse: OFF",
                fg_color="#555", hover_color="#777"
            )

    def _on_cooldown_changed(self, value):
        """Handle reaction speed slider change."""
        ms = int(value)
        self._cooldown_label.configure(text=f"{ms} ms")
        if self.command_executor:
            self.command_executor.set_cooldown(ms)

    def _handle_logout(self):
        """Handle user logout."""
        self._stop_camera()
        self.auth.logout()
        self._build_login_screen()

    def _on_close(self):
        """Handle window close."""
        self._running = False
        self._stop_camera()
        if self.db:
            self.db.close()
        self.destroy()


def main():
    """Application entry point."""
    logger.info(f"Starting {APP_TITLE} v{APP_VERSION}")
    
    try:
        app = GestureAIApp()
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Application shutdown")


if __name__ == "__main__":
    main()
