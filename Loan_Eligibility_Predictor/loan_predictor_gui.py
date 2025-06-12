import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
from loan_predictor import LoanEligibilityPredictor
import joblib
import os
import threading
import time

class ModernButton(tk.Button):
    """Custom modern button with hover effects"""
    def __init__(self, parent, **kwargs):
        # Extract custom parameters before passing to parent
        self.default_bg = kwargs.get('bg', '#3498db')
        self.hover_bg = kwargs.pop('hover_bg', '#2980b9')  # Remove from kwargs
        
        # Initialize parent button
        super().__init__(parent, **kwargs)
        
        # Bind hover events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        
    def on_enter(self, e):
        self.config(bg=self.hover_bg)
        
    def on_leave(self, e):
        self.config(bg=self.default_bg)

class AnimatedProgressBar:
    """Custom animated progress bar"""
    def __init__(self, parent, width=300, height=20):
        self.canvas = tk.Canvas(parent, width=width, height=height, bg='white', highlightthickness=0)
        self.width = width
        self.height = height
        self.progress = 0
        self.is_running = False
        
    def start(self):
        self.is_running = True
        self.progress = 0
        self.animate()
        
    def stop(self):
        self.is_running = False
        self.canvas.delete("all")
        
    def animate(self):
        if self.is_running:
            self.canvas.delete("all")
            # Background
            self.canvas.create_rectangle(0, 0, self.width, self.height, 
                                       fill='#ecf0f1', outline='#bdc3c7')
            # Progress bar
            progress_width = (self.progress / 100) * self.width
            self.canvas.create_rectangle(0, 0, progress_width, self.height, 
                                       fill='#3498db', outline='')
            
            # Animated gradient effect
            for i in range(0, int(progress_width), 2):
                alpha = 1 - (i / progress_width) if progress_width > 0 else 0
                color_intensity = int(255 * alpha)
                color = f'#{52 + color_intensity//4:02x}{152 + color_intensity//8:02x}{219:02x}'
                self.canvas.create_line(i, 0, i, self.height, fill=color)
            
            self.progress = (self.progress + 2) % 100
            self.canvas.after(50, self.animate)

class LoanPredictorGUI:
    """
    Modern Tkinter GUI application for the Loan Eligibility Predictor.
    Features enhanced UI/UX with animations, modern styling, and intuitive design.
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🏦 Smart Loan Predictor AI")
        self.root.geometry("1000x800")
        self.root.configure(bg='#f8f9fa')
        
        # Modern color scheme
        self.colors = {
            'primary': '#3498db',
            'success': '#2ecc71',
            'danger': '#e74c3c',
            'warning': '#f39c12',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#2c3e50',
            'secondary': '#6c757d'
        }
        
        # Initialize predictor
        self.predictor = LoanEligibilityPredictor()
        self.model_loaded = False
        self.training_thread = None
        
        # Setup styles
        self.setup_styles()
        
        # Create GUI elements
        self.create_modern_gui()
        
        # Try to load existing model
        self.load_existing_model()
        
    def setup_styles(self):
        """Setup modern ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure modern notebook style
        style.configure('Modern.TNotebook', 
                       background=self.colors['light'],
                       borderwidth=0)
        style.configure('Modern.TNotebook.Tab',
                       background=self.colors['secondary'],
                       foreground='white',
                       padding=[20, 10],
                       font=('Segoe UI', 10, 'bold'))
        style.map('Modern.TNotebook.Tab',
                 background=[('selected', self.colors['primary']),
                           ('active', self.colors['info'])])
        
        # Configure modern frame style
        style.configure('Card.TFrame',
                       background='white',
                       relief='flat',
                       borderwidth=1)
        
        # Configure modern labelframe style
        style.configure('Card.TLabelframe',
                       background='white',
                       foreground=self.colors['dark'],
                       font=('Segoe UI', 11, 'bold'))
        
    def create_modern_gui(self):
        """Create modern GUI with enhanced styling"""
        # Header
        self.create_header()
        
        # Main container
        main_container = tk.Frame(self.root, bg=self.colors['light'])
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Create modern notebook
        self.notebook = ttk.Notebook(main_container, style='Modern.TNotebook')
        self.notebook.pack(fill='both', expand=True)
        
        # Create tabs with icons
        self.create_prediction_tab_modern()
        self.create_training_tab_modern()
        self.create_analytics_tab_modern()
        self.create_help_tab()
        
    def create_header(self):
        """Create modern header with gradient effect"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        header_frame.pack(fill='x')
        header_frame.pack_propagate(False)
        
        # Title with shadow effect
        title_container = tk.Frame(header_frame, bg=self.colors['primary'])
        title_container.pack(expand=True)
        
        # Main title
        main_title = tk.Label(
            title_container,
            text="🏦 Smart Loan Predictor AI",
            font=('Segoe UI', 24, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        main_title.pack(pady=10)
        
        # Subtitle
        subtitle = tk.Label(
            title_container,
            text="Intelligent Loan Eligibility Assessment with Machine Learning",
            font=('Segoe UI', 10),
            bg=self.colors['primary'],
            fg='#ecf0f1'
        )
        subtitle.pack()
        
        # Status indicator
        self.status_indicator = tk.Label(
            header_frame,
            text="● System Ready",
            font=('Segoe UI', 9),
            bg=self.colors['primary'],
            fg='#2ecc71'
        )
        self.status_indicator.pack(side='right', padx=20, pady=5)
        
    def create_prediction_tab_modern(self):
        """Create modern prediction tab with card-based layout"""
        pred_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(pred_frame, text="🔍 Quick Prediction")
        
        # Main scroll container
        canvas = tk.Canvas(pred_frame, bg=self.colors['light'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(pred_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['light'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Welcome card
        welcome_card = tk.Frame(scrollable_frame, bg='white', relief='raised', bd=1)
        welcome_card.pack(fill='x', padx=20, pady=15)
        
        welcome_label = tk.Label(
            welcome_card,
            text="💡 Enter Your Loan Application Details",
            font=('Segoe UI', 16, 'bold'),
            bg='white',
            fg=self.colors['dark']
        )
        welcome_label.pack(pady=15)
        
        description = tk.Label(
            welcome_card,
            text="Our AI will analyze your information and provide an instant loan eligibility decision",
            font=('Segoe UI', 10),
            bg='white',
            fg=self.colors['secondary'],
            wraplength=600
        )
        description.pack(pady=(0, 15))
        
        # Input cards container
        cards_container = tk.Frame(scrollable_frame, bg=self.colors['light'])
        cards_container.pack(fill='x', padx=20, pady=10)
        
        # Personal Information Card
        self.create_input_card(cards_container, "👤 Personal Information", 
                              self.create_personal_inputs, row=0, column=0)
        
        # Financial Information Card
        self.create_input_card(cards_container, "💰 Financial Information", 
                              self.create_financial_inputs, row=0, column=1)
        
        # Employment & Property Card
        self.create_input_card(cards_container, "🏢 Employment & Property", 
                              self.create_employment_inputs, row=1, column=0, columnspan=2)
        
        # Prediction button
        self.create_prediction_button(scrollable_frame)
        
        # Results card
        self.create_results_card(scrollable_frame)
        
        # Pack canvas
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def create_input_card(self, parent, title, content_func, row, column, columnspan=1):
        """Create a modern input card"""
        card = tk.Frame(parent, bg='white', relief='raised', bd=1)
        card.grid(row=row, column=column, columnspan=columnspan, 
                 sticky='ew', padx=10, pady=10)
        
        # Card header
        header = tk.Frame(card, bg=self.colors['primary'], height=40)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title_label = tk.Label(
            header,
            text=title,
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        )
        title_label.pack(expand=True)
        
        # Card content
        content_frame = tk.Frame(card, bg='white')
        content_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        content_func(content_frame)
        
        # Configure grid weights
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        
    def create_personal_inputs(self, parent):
        """Create personal information inputs"""
        self.input_vars = getattr(self, 'input_vars', {})
        
        # Age
        self.create_modern_input(parent, "Age", "30", 'spinbox', 
                               from_=18, to=70, row=0)
        
        # Education
        self.create_modern_input(parent, "Education Level", "Bachelor", 'combobox',
                               values=['High School', 'Bachelor', 'Master', 'PhD'], row=1)
        
        # Marital Status
        self.create_modern_input(parent, "Marital Status", "Single", 'combobox',
                               values=['Single', 'Married', 'Divorced'], row=2)
        
    def create_financial_inputs(self, parent):
        """Create financial information inputs"""
        # Income
        self.create_modern_input(parent, "Annual Income ($)", "50000", 'entry', row=0)
        
        # Credit Score
        self.create_modern_input(parent, "Credit Score", "650", 'spinbox',
                               from_=300, to=850, row=1)
        
        # Loan Amount
        self.create_modern_input(parent, "Loan Amount ($)", "200000", 'entry', row=2)
        
    def create_employment_inputs(self, parent):
        """Create employment and property inputs"""
        # Employment Years
        self.create_modern_input(parent, "Employment Years", "5", 'spinbox',
                               from_=0, to=40, row=0, column=0)
        
        # Employment Status
        self.create_modern_input(parent, "Employment Status", "Employed", 'combobox',
                               values=['Employed', 'Self-Employed', 'Unemployed'], 
                               row=0, column=1)
        
        # Property Area
        self.create_modern_input(parent, "Property Area", "Urban", 'combobox',
                               values=['Urban', 'Semi-Urban', 'Rural'], 
                               row=0, column=2)
        
        # Configure grid
        for i in range(3):
            parent.grid_columnconfigure(i, weight=1)
            
    def create_modern_input(self, parent, label_text, default_value, widget_type, 
                          row=0, column=0, **kwargs):
        """Create a modern input field"""
        container = tk.Frame(parent, bg='white')
        container.grid(row=row, column=column, sticky='ew', padx=5, pady=8)
        
        # Label
        label = tk.Label(
            container,
            text=label_text,
            font=('Segoe UI', 9, 'bold'),
            bg='white',
            fg=self.colors['dark']
        )
        label.pack(anchor='w')
        
        # Input variable
        var_name = label_text.replace(' ', '_').replace('(', '').replace(')', '').replace('$', '')
        if var_name not in self.input_vars:
            self.input_vars[var_name] = tk.StringVar(value=default_value)
        
        # Widget
        if widget_type == 'entry':
            widget = tk.Entry(
                container,
                textvariable=self.input_vars[var_name],
                font=('Segoe UI', 10),
                relief='flat',
                bd=1,
                bg='#f8f9fa'
            )
        elif widget_type == 'spinbox':
            widget = tk.Spinbox(
                container,
                textvariable=self.input_vars[var_name],
                font=('Segoe UI', 10),
                relief='flat',
                bd=1,
                bg='#f8f9fa',
                **kwargs
            )
        elif widget_type == 'combobox':
            widget = ttk.Combobox(
                container,
                textvariable=self.input_vars[var_name],
                font=('Segoe UI', 10),
                state='readonly',
                **kwargs
            )
            
        widget.pack(fill='x', pady=(2, 0))
        
    def create_prediction_button(self, parent):
        """Create animated prediction button"""
        button_container = tk.Frame(parent, bg=self.colors['light'])
        button_container.pack(pady=30)
        
        self.predict_btn = ModernButton(
            button_container,
            text="🚀 Analyze Loan Eligibility",
            command=self.predict_loan_animated,
            bg=self.colors['success'],
            hover_bg='#27ae60',
            fg='white',
            font=('Segoe UI', 14, 'bold'),
            padx=40,
            pady=15,
            relief='flat',
            cursor='hand2'
        )
        self.predict_btn.pack()
        
        # Progress bar (hidden initially)
        self.progress_bar = AnimatedProgressBar(button_container)
        
    def create_results_card(self, parent):
        """Create modern results display card"""
        self.results_card = tk.Frame(parent, bg='white', relief='raised', bd=1)
        self.results_card.pack(fill='x', padx=20, pady=20)
        
        # Results header
        results_header = tk.Frame(self.results_card, bg=self.colors['info'], height=40)
        results_header.pack(fill='x')
        results_header.pack_propagate(False)
        
        header_label = tk.Label(
            results_header,
            text="📊 Analysis Results",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['info'],
            fg='white'
        )
        header_label.pack(expand=True)
        
        # Results content
        self.results_content = tk.Frame(self.results_card, bg='white')
        self.results_content.pack(fill='both', expand=True, padx=20, pady=20)
        
        self.result_display = tk.Label(
            self.results_content,
            text="🎯 Ready for Analysis\n\nComplete the form above and click 'Analyze Loan Eligibility' to get your instant decision!",
            font=('Segoe UI', 11),
            bg='white',
            fg=self.colors['secondary'],
            justify='center',
            wraplength=700
        )
        self.result_display.pack(expand=True)
        
    def create_training_tab_modern(self):
        """Create modern training tab"""
        train_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(train_frame, text="🤖 AI Training")
        
        # Training card
        training_card = tk.Frame(train_frame, bg='white', relief='raised', bd=1)
        training_card.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(training_card, bg=self.colors['warning'], height=50)
        header.pack(fill='x')
        header.pack_propagate(False)
        
        title = tk.Label(
            header,
            text="🧠 Train Your AI Model",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['warning'],
            fg='white'
        )
        title.pack(expand=True)
        
        # Content
        content = tk.Frame(training_card, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=20)
        
        # Description
        desc = tk.Label(
            content,
            text="Train machine learning models on synthetic loan data to improve prediction accuracy",
            font=('Segoe UI', 12),
            bg='white',
            fg=self.colors['dark'],
            wraplength=600
        )
        desc.pack(pady=(0, 20))
        
        # Controls
        controls_frame = tk.Frame(content, bg='white')
        controls_frame.pack(fill='x', pady=10)
        
        tk.Label(
            controls_frame,
            text="Training Samples:",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg=self.colors['dark']
        ).pack(side='left', padx=(0, 10))
        
        self.samples_var = tk.StringVar(value="1000")
        samples_spinner = tk.Spinbox(
            controls_frame,
            from_=100, to=10000, increment=100,
            textvariable=self.samples_var,
            font=('Segoe UI', 10),
            width=10,
            relief='flat',
            bd=1
        )
        samples_spinner.pack(side='left', padx=(0, 20))
        
        self.train_btn = ModernButton(
            controls_frame,
            text="🚀 Start Training",
            command=self.train_models_threaded,
            bg=self.colors['danger'],
            hover_bg='#c0392b',
            fg='white',
            font=('Segoe UI', 12, 'bold'),
            padx=30,
            pady=10,
            relief='flat',
            cursor='hand2'
        )
        self.train_btn.pack(side='left')
        
        # Progress indicator
        self.training_progress = AnimatedProgressBar(content, width=400, height=25)
        self.training_progress.canvas.pack(pady=20)
        
        # Training log
        log_frame = tk.Frame(content, bg='white')
        log_frame.pack(fill='both', expand=True, pady=(10, 0))
        
        tk.Label(
            log_frame,
            text="📝 Training Log",
            font=('Segoe UI', 11, 'bold'),
            bg='white',
            fg=self.colors['dark']
        ).pack(anchor='w', pady=(0, 10))
        
        # Create text widget with modern styling
        text_frame = tk.Frame(log_frame, relief='sunken', bd=1)
        text_frame.pack(fill='both', expand=True)
        
        self.training_log = tk.Text(
            text_frame,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg=self.colors['dark'],
            relief='flat',
            wrap='word'
        )
        
        log_scrollbar = ttk.Scrollbar(text_frame, command=self.training_log.yview)
        self.training_log.configure(yscrollcommand=log_scrollbar.set)
        
        self.training_log.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
    def create_analytics_tab_modern(self):
        """Create modern analytics and results tab"""
        analytics_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(analytics_frame, text="📈 Analytics")
        
        # Status card
        status_card = tk.Frame(analytics_frame, bg='white', relief='raised', bd=1)
        status_card.pack(fill='x', padx=20, pady=20)
        
        # Status header
        status_header = tk.Frame(status_card, bg=self.colors['info'], height=40)
        status_header.pack(fill='x')
        status_header.pack_propagate(False)
        
        tk.Label(
            status_header,
            text="🔧 System Status",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['info'],
            fg='white'
        ).pack(expand=True)
        
        # Status content
        status_content = tk.Frame(status_card, bg='white')
        status_content.pack(fill='both', padx=20, pady=15)
        
        self.status_display = tk.Label(
            status_content,
            text="⚠️ No model loaded. Please train a model first.",
            font=('Segoe UI', 11),
            bg='white',
            fg=self.colors['warning']
        )
        self.status_display.pack()
        
        # Model management
        mgmt_card = tk.Frame(analytics_frame, bg='white', relief='raised', bd=1)
        mgmt_card.pack(fill='x', padx=20, pady=(0, 20))
        
        mgmt_header = tk.Frame(mgmt_card, bg=self.colors['secondary'], height=40)
        mgmt_header.pack(fill='x')
        mgmt_header.pack_propagate(False)
        
        tk.Label(
            mgmt_header,
            text="💾 Model Management",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['secondary'],
            fg='white'
        ).pack(expand=True)
        
        # Management buttons
        mgmt_content = tk.Frame(mgmt_card, bg='white')
        mgmt_content.pack(fill='x', padx=20, pady=15)
        
        button_frame = tk.Frame(mgmt_content, bg='white')
        button_frame.pack()
        
        ModernButton(
            button_frame,
            text="📁 Load Model",
            command=self.load_model_file,
            bg=self.colors['warning'],
            hover_bg='#e67e22',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=8,
            relief='flat'
        ).pack(side='left', padx=(0, 10))
        
        ModernButton(
            button_frame,
            text="💾 Save Model",
            command=self.save_model_file,
            bg=self.colors['success'],
            hover_bg='#27ae60',
            fg='white',
            font=('Segoe UI', 10, 'bold'),
            padx=20,
            pady=8,
            relief='flat'
        ).pack(side='left')
        
        # Analytics display
        analytics_card = tk.Frame(analytics_frame, bg='white', relief='raised', bd=1)
        analytics_card.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        analytics_header = tk.Frame(analytics_card, bg=self.colors['primary'], height=40)
        analytics_header.pack(fill='x')
        analytics_header.pack_propagate(False)
        
        tk.Label(
            analytics_header,
            text="📊 Model Analytics",
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['primary'],
            fg='white'
        ).pack(expand=True)
        
        # Analytics content
        analytics_content = tk.Frame(analytics_card, bg='white')
        analytics_content.pack(fill='both', expand=True, padx=20, pady=15)
        
        text_frame = tk.Frame(analytics_content, relief='sunken', bd=1)
        text_frame.pack(fill='both', expand=True)
        
        self.analytics_display = tk.Text(
            text_frame,
            font=('Consolas', 9),
            bg='#f8f9fa',
            fg=self.colors['dark'],
            relief='flat',
            wrap='word'
        )
        
        analytics_scrollbar = ttk.Scrollbar(text_frame, command=self.analytics_display.yview)
        self.analytics_display.configure(yscrollcommand=analytics_scrollbar.set)
        
        self.analytics_display.pack(side='left', fill='both', expand=True)
        analytics_scrollbar.pack(side='right', fill='y')
        
    def create_help_tab(self):
        """Create help and information tab"""
        help_frame = tk.Frame(self.notebook, bg=self.colors['light'])
        self.notebook.add(help_frame, text="❓ Help")
        
        # Help content
        help_card = tk.Frame(help_frame, bg='white', relief='raised', bd=1)
        help_card.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Header
        header = tk.Frame(help_card, bg=self.colors['info'], height=50)
        header.pack(fill='x')
        
        tk.Label(
            header,
            text="📚 How to Use Smart Loan Predictor AI",
            font=('Segoe UI', 16, 'bold'),
            bg=self.colors['info'],
            fg='white'
        ).pack(expand=True)
        
        # Content
        content = tk.Frame(help_card, bg='white')
        content.pack(fill='both', expand=True, padx=30, pady=20)
        
        help_text = """
🔍 QUICK PREDICTION:
• Fill in your personal, financial, and employment details
• Click "Analyze Loan Eligibility" for instant results
• View your approval probability and risk assessment

🤖 AI TRAINING:
• Train machine learning models on synthetic data
• Adjust the number of training samples (100-10,000)
• Compare Logistic Regression vs Random Forest performance
• Models are automatically saved after training

📈 ANALYTICS:
• View detailed model performance metrics
• Load previously trained models
• Save your best performing models
• Monitor system status

🎯 TIPS FOR BETTER PREDICTIONS:
• Higher credit scores improve approval chances
• Stable employment history is favorable
• Lower loan-to-income ratios are preferred
• Urban properties typically have better approval rates

⚠️ IMPORTANT NOTES:
• This tool uses synthetic data for demonstration
• Real loan decisions involve many additional factors
• Always consult with financial institutions for actual loans
• Predictions are for educational purposes only
        """
        
        help_label = tk.Label(
            content,
            text=help_text,
            font=('Segoe UI', 10),
            bg='white',
            fg=self.colors['dark'],
            justify='left',
            wraplength=700
        )
        help_label.pack(anchor='w')
        
    def predict_loan_animated(self):
        """Animated loan prediction with progress indication"""
        if not self.model_loaded:
            messagebox.showerror("Model Required", 
                               "🚫 No AI model loaded!\n\nPlease train a model first or load an existing one.")
            return
            
        # Start animation
        self.predict_btn.config(state='disabled', text="🔄 Analyzing...")
        self.progress_bar.canvas.pack(pady=10)
        self.progress_bar.start()
        
        # Update results to show processing
        self.result_display.config(
            text="🔄 AI Analysis in Progress...\n\nProcessing your loan application data...",
            fg=self.colors['info']
        )
        
        # Run prediction in thread to avoid blocking UI
        def run_prediction():
            time.sleep(2)  # Simulate processing time
            self.root.after(0, self.complete_prediction)
            
        threading.Thread(target=run_prediction, daemon=True).start()
        
    def complete_prediction(self):
        """Complete the prediction process"""
        try:
            # Collect input data
            application_data = {
                'Age': int(self.input_vars['Age'].get()),
                'Income': float(self.input_vars['Annual_Income_'].get()),
                'Credit_Score': int(self.input_vars['Credit_Score'].get()),
                'Employment_Years': int(self.input_vars['Employment_Years'].get()),
                'Loan_Amount': float(self.input_vars['Loan_Amount_'].get()),
                'Education': self.input_vars['Education_Level'].get(),
                'Marital_Status': self.input_vars['Marital_Status'].get(),
                'Employment_Status': self.input_vars['Employment_Status'].get(),
                'Property_Area': self.input_vars['Property_Area'].get()
            }
            
            # Make prediction
            prediction, probability = self.predictor.predict_single_application(application_data)
            
            # Stop animation
            self.progress_bar.stop()
            self.progress_bar.canvas.pack_forget()
            self.predict_btn.config(state='normal', text="🚀 Analyze Loan Eligibility")
            
            # Display results with enhanced formatting
            self.display_prediction_results(prediction, probability, application_data)
            
        except ValueError:
            self.progress_bar.stop()
            self.progress_bar.canvas.pack_forget()
            self.predict_btn.config(state='normal', text="🚀 Analyze Loan Eligibility")
            messagebox.showerror("Input Error", 
                               "❌ Invalid Input Data\n\nPlease ensure all numerical fields contain valid numbers.")
        except Exception as e:
            self.progress_bar.stop()
            self.progress_bar.canvas.pack_forget()
            self.predict_btn.config(state='normal', text="🚀 Analyze Loan Eligibility")
            messagebox.showerror("Prediction Error", f"❌ Analysis Failed\n\n{str(e)}")
            
    def display_prediction_results(self, prediction, probability, application_data):
        """Display prediction results with modern styling"""
        # Clear previous results
        for widget in self.results_content.winfo_children():
            widget.destroy()
            
        # Create results layout
        results_container = tk.Frame(self.results_content, bg='white')
        results_container.pack(fill='both', expand=True)
        
        # Main result
        if prediction == 1:
            result_icon = "🎉"
            result_text = "LOAN APPROVED!"
            result_color = self.colors['success']
            bg_color = '#d5f4e6'
        else:
            result_icon = "❌"
            result_text = "LOAN REJECTED"
            result_color = self.colors['danger']
            bg_color = '#faeaea'
            
        # Result header
        result_header = tk.Frame(results_container, bg=bg_color, height=80)
        result_header.pack(fill='x', pady=(0, 20))
        result_header.pack_propagate(False)
        
        main_result = tk.Label(
            result_header,
            text=f"{result_icon} {result_text}",
            font=('Segoe UI', 20, 'bold'),
            bg=bg_color,
            fg=result_color
        )
        main_result.pack(expand=True)
        
        # Confidence score
        confidence_frame = tk.Frame(results_container, bg='white')
        confidence_frame.pack(fill='x', pady=10)
        
        tk.Label(
            confidence_frame,
            text="Confidence Score:",
            font=('Segoe UI', 12, 'bold'),
            bg='white',
            fg=self.colors['dark']
        ).pack(side='left')
        
        confidence_bar = tk.Canvas(confidence_frame, width=200, height=20, 
                                 bg='#ecf0f1', highlightthickness=0)
        confidence_bar.pack(side='left', padx=10)
        
        # Draw confidence bar
        bar_width = int(200 * probability)
        confidence_bar.create_rectangle(0, 0, bar_width, 20, 
                                      fill=result_color, outline='')
        
        tk.Label(
            confidence_frame,
            text=f"{probability:.1%}",
            font=('Segoe UI', 12, 'bold'),
            bg='white',
            fg=result_color
        ).pack(side='left', padx=10)
        
        # Risk assessment
        risk_frame = tk.Frame(results_container, bg='white')
        risk_frame.pack(fill='x', pady=10)
        
        if probability >= 0.8:
            risk_level = "Very Low Risk 🟢"
            risk_color = self.colors['success']
        elif probability >= 0.6:
            risk_level = "Low Risk 🟡"
            risk_color = self.colors['warning']
        elif probability >= 0.4:
            risk_level = "Medium Risk 🟠"
            risk_color = '#fd9644'
        elif probability >= 0.2:
            risk_level = "High Risk 🔴"
            risk_color = self.colors['danger']
        else:
            risk_level = "Very High Risk ⚫"
            risk_color = '#6c757d'
            
        tk.Label(
            risk_frame,
            text=f"Risk Assessment: {risk_level}",
            font=('Segoe UI', 12, 'bold'),
            bg='white',
            fg=risk_color
        ).pack()
        
        # Application summary
        summary_frame = tk.Frame(results_container, bg='#f8f9fa', relief='raised', bd=1)
        summary_frame.pack(fill='x', pady=20)
        
        tk.Label(
            summary_frame,
            text="📋 Application Summary",
            font=('Segoe UI', 12, 'bold'),
            bg='#f8f9fa',
            fg=self.colors['dark']
        ).pack(pady=10)
        
        # Create summary in two columns
        summary_content = tk.Frame(summary_frame, bg='#f8f9fa')
        summary_content.pack(fill='x', padx=20, pady=(0, 15))
        
        left_col = tk.Frame(summary_content, bg='#f8f9fa')
        left_col.pack(side='left', fill='both', expand=True)
        
        right_col = tk.Frame(summary_content, bg='#f8f9fa')
        right_col.pack(side='right', fill='both', expand=True)
        
        # Left column data
        left_data = [
            ("👤 Age", f"{application_data['Age']} years"),
            ("💰 Income", f"${application_data['Income']:,.0f}"),
            ("📊 Credit Score", f"{application_data['Credit_Score']}"),
            ("💼 Employment", f"{application_data['Employment_Years']} years"),
            ("🏠 Loan Amount", f"${application_data['Loan_Amount']:,.0f}")
        ]
        
        for label, value in left_data:
            row = tk.Frame(left_col, bg='#f8f9fa')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, font=('Segoe UI', 9), 
                    bg='#f9f9fa', fg=self.colors['secondary']).pack(side='left')
            tk.Label(row, text=value, font=('Segoe UI', 9, 'bold'), 
                    bg='#f9f9fa', fg=self.colors['dark']).pack(side='right')
        
        # Right column data
        right_data = [
            ("🎓 Education", application_data['Education']),
            ("💑 Marital Status", application_data['Marital_Status']),
            ("🏢 Employment", application_data['Employment_Status']),
            ("🏘️ Property Area", application_data['Property_Area']),
            ("📈 Debt-to-Income", f"{(application_data['Loan_Amount']/application_data['Income']):.1f}x")
        ]
        
        for label, value in right_data:
            row = tk.Frame(right_col, bg='#f8f9fa')
            row.pack(fill='x', pady=2)
            tk.Label(row, text=label, font=('Segoe UI', 9), 
                    bg='#f9f9fa', fg=self.colors['secondary']).pack(side='left')
            tk.Label(row, text=value, font=('Segoe UI', 9, 'bold'), 
                    bg='#f9f9fa', fg=self.colors['dark']).pack(side='right')
        
    def train_models_threaded(self):
        """Train models in a separate thread"""
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showinfo("Training in Progress", "🔄 Model training is already in progress!")
            return
            
        self.training_thread = threading.Thread(target=self.train_models_with_progress, daemon=True)
        self.training_thread.start()
        
    def train_models_with_progress(self):
        """Train models with progress indication"""
        try:
            # Update UI on main thread
            self.root.after(0, lambda: self.train_btn.config(state='disabled', text="🔄 Training..."))
            self.root.after(0, lambda: self.training_progress.start())
            self.root.after(0, lambda: self.training_log.delete(1.0, tk.END))
            
            n_samples = int(self.samples_var.get())
            
            # Training steps with progress updates
            steps = [
                (f"🚀 Initializing training with {n_samples} samples...\n", self.init_training),
                ("📊 Generating synthetic loan data...\n", lambda: self.predictor.load_and_explore_data(
                    self.predictor.generate_synthetic_data(n_samples))),
                ("🔄 Preprocessing data...\n", self.predictor.preprocess_data),
                ("🤖 Training AI models...\n", self.predictor.train_models),
                ("📈 Evaluating performance...\n", self.predictor.evaluate_models),
                ("💾 Saving trained model...\n", lambda: self.predictor.save_model())
            ]
            
            results = None
            for i, (message, func) in enumerate(steps):
                self.root.after(0, lambda m=message: self.log_message(m))
                time.sleep(0.5)  # Brief pause for UI update
                
                if func == self.predictor.evaluate_models:
                    results = func()
                else:
                    func()
                    
                time.sleep(0.5)  # Simulate processing time
                
            # Complete training
            self.root.after(0, lambda: self.complete_training(results))
            
        except Exception as e:
            self.root.after(0, lambda: self.training_failed(str(e)))
            
    def init_training(self):
        """Initialize training process"""
        pass
        
    def log_message(self, message):
        """Add message to training log"""
        self.training_log.insert(tk.END, message)
        self.training_log.see(tk.END)
        self.training_log.update()
        
    def complete_training(self, results):
        """Complete the training process"""
        self.training_progress.stop()
        self.train_btn.config(state='normal', text="🚀 Start Training")
        
        # Log results
        self.log_message("\n" + "="*50 + "\n")
        self.log_message("🎉 TRAINING COMPLETED!\n")
        self.log_message("="*50 + "\n\n")
        
        if results:
            for model_name, metrics in results.items():
                self.log_message(f"📊 {model_name}:\n")
                self.log_message(f"   Training Accuracy: {metrics['train_accuracy']:.4f}\n")
                self.log_message(f"   Test Accuracy: {metrics['test_accuracy']:.4f}\n")
                self.log_message(f"   CV Score: {metrics['cv_mean']:.4f} ± {metrics['cv_std']*2:.4f}\n\n")
                
            # Determine best model
            if results['Random Forest']['test_accuracy'] > results['Logistic Regression']['test_accuracy']:
                best_model = 'Random Forest'
                best_accuracy = results['Random Forest']['test_accuracy']
            else:
                best_model = 'Logistic Regression'
                best_accuracy = results['Logistic Regression']['test_accuracy']
                
            self.log_message(f"🏆 Best Model: {best_model}\n")
            self.log_message(f"🎯 Best Accuracy: {best_accuracy:.4f}\n")
            
        self.log_message("\n✅ Model ready for predictions!\n")
        
        # Update status
        self.model_loaded = True
        self.update_status(f"✅ AI Model Trained Successfully! Best: {best_model} ({best_accuracy:.3f})")
        self.update_analytics(results)
        
        # Show completion message
        messagebox.showinfo("Training Complete", 
                          f"🎉 AI Training Successful!\n\n"
                          f"Best Model: {best_model}\n"
                          f"Accuracy: {best_accuracy:.1%}\n\n"
                          f"Your model is now ready for predictions!")
        
    def training_failed(self, error):
        """Handle training failure"""
        self.training_progress.stop()
        self.train_btn.config(state='normal', text="🚀 Start Training")
        self.log_message(f"\n❌ TRAINING FAILED!\n")
        self.log_message(f"Error: {error}\n")
        messagebox.showerror("Training Failed", f"❌ AI Training Failed\n\n{error}")
        
    def load_existing_model(self):
        """Try to load an existing model file"""
        if os.path.exists('loan_predictor_model.pkl'):
            try:
                self.predictor.load_model('loan_predictor_model.pkl')
                self.model_loaded = True
                self.update_status("✅ Pre-trained AI model loaded successfully!")
                self.update_analytics_simple("Pre-trained model loaded from loan_predictor_model.pkl")
                self.status_indicator.config(text="● AI Model Ready", fg=self.colors['success'])
            except Exception as e:
                self.update_status(f"❌ Error loading model: {str(e)}")
                self.status_indicator.config(text="● System Error", fg=self.colors['danger'])
        else:
            self.update_status("⚠️ No pre-trained model found. Please train a model first.")
            self.status_indicator.config(text="● Training Required", fg=self.colors['warning'])
            
    def load_model_file(self):
        """Load a model from file with modern dialog"""
        filename = filedialog.askopenfilename(
            title="Load AI Model",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if filename:
            try:
                self.predictor.load_model(filename)
                self.model_loaded = True
                model_name = os.path.basename(filename)
                self.update_status(f"✅ Model loaded: {model_name}")
                self.update_analytics_simple(f"Model loaded from: {filename}")
                self.status_indicator.config(text="● AI Model Ready", fg=self.colors['success'])
                messagebox.showinfo("Success", f"✅ Model Loaded Successfully\n\n{model_name}")
            except Exception as e:
                messagebox.showerror("Load Error", f"❌ Failed to Load Model\n\n{str(e)}")
                
    def save_model_file(self):
        """Save the current model to file"""
        if not self.model_loaded:
            messagebox.showerror("No Model", "❌ No Model to Save\n\nPlease train a model first.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save AI Model",
            defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All files", "*.*")],
            initialdir=os.getcwd()
        )
        
        if filename:
            try:
                self.predictor.save_model(filename)
                model_name = os.path.basename(filename)
                messagebox.showinfo("Success", f"✅ Model Saved Successfully\n\n{model_name}")
            except Exception as e:
                messagebox.showerror("Save Error", f"❌ Failed to Save Model\n\n{str(e)}")
                
    def update_status(self, message):
        """Update the status display with modern styling"""
        if "✅" in message:
            color = self.colors['success']
        elif "❌" in message:
            color = self.colors['danger']
        elif "⚠️" in message:
            color = self.colors['warning']
        else:
            color = self.colors['dark']
            
        self.status_display.config(text=message, fg=color)
        
    def update_analytics(self, results):
        """Update analytics display with training results"""
        self.analytics_display.delete(1.0, tk.END)
        
        if results:
            analytics_text = "🤖 AI MODEL TRAINING RESULTS\n"
            analytics_text += "=" * 50 + "\n\n"
            
            analytics_text += f"📊 Training Data: {self.samples_var.get()} samples\n"
            analytics_text += f"🕒 Training Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            analytics_text += "📈 MODEL PERFORMANCE:\n"
            analytics_text += "-" * 30 + "\n\n"
            
            for model_name, metrics in results.items():
                analytics_text += f"🔹 {model_name}:\n"
                analytics_text += f"   Training Accuracy: {metrics['train_accuracy']:.4f} ({metrics['train_accuracy']*100:.2f}%)\n"
                analytics_text += f"   Test Accuracy:     {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)\n"
                analytics_text += f"   Cross-Validation:  {metrics['cv_mean']:.4f} ± {metrics['cv_std']*2:.4f}\n\n"
                
            # Determine best model
            if results['Random Forest']['test_accuracy'] > results['Logistic Regression']['test_accuracy']:
                best_model = 'Random Forest'
                best_accuracy = results['Random Forest']['test_accuracy']
            else:
                best_model = 'Logistic Regression'
                best_accuracy = results['Logistic Regression']['test_accuracy']
                
            analytics_text += f"🏆 BEST MODEL: {best_model}\n"
            analytics_text += f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n\n"
            
            analytics_text += "💡 MODEL INSIGHTS:\n"
            analytics_text += "-" * 20 + "\n"
            analytics_text += "• Higher accuracy indicates better prediction capability\n"
            analytics_text += "• Cross-validation score shows model consistency\n"
            analytics_text += "• Random Forest typically handles complex patterns better\n"
            analytics_text += "• Logistic Regression is more interpretable\n\n"
            
            analytics_text += "✅ Model is ready for loan predictions!"
            
        else:
            analytics_text = "⚠️ No training results available.\n\nPlease train a model first to view analytics."
            
        self.analytics_display.insert(1.0, analytics_text)
        
    def update_analytics_simple(self, message):
        """Update analytics with simple message"""
        self.analytics_display.delete(1.0, tk.END)
        self.analytics_display.insert(1.0, message)

def main():
    """Main function to run the modern GUI application"""
    root = tk.Tk()
    
    # Set window icon and properties
    root.resizable(True, True)
    root.minsize(800, 600)
    
    # Create and run app
    app = LoanPredictorGUI(root)
    
    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()