"""
SmartCAPEX AI - Main Application Window
======================================

Main GUI class for the SmartCAPEX AI desktop application
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import threading
import queue
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.surrogate_modeler import SurrogateModeler
from agents.multi_objective_optimizer import MultiObjectiveOptimizer
from agents.climate_guardian import ClimateGuardian
from agents.asset_manager import AssetManager


class SmartCAPEXMainWindow:
    """
    Main application window for SmartCAPEX AI
    
    Integrates all four agents:
    1. Surrogate Modeler (EANN Core)
    2. Multi-Objective Optimizer
    3. Climate Guardian
    4. Asset Manager
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("SmartCAPEX AI - Maritime Retrofit Decision Support")
        
        # Initialize agents
        self.surrogate_modeler = SurrogateModeler()
        self.optimizer = MultiObjectiveOptimizer()
        self.climate_guardian = ClimateGuardian()
        self.asset_manager = AssetManager()
        
        # Data storage
        self.current_vessel_data = {}
        self.analysis_results = {}
        self.is_model_trained = False
        
        # Create GUI components
        self.create_styles()
        self.create_menu()
        self.create_main_layout()
        self.create_input_panel()
        self.create_results_panel()
        self.create_status_bar()
        
        # Load default vessel data
        self.load_default_data()
    
    def create_styles(self):
        """Create custom styles for the application"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#2C3E50')
        self.style.configure('Section.TLabel', font=('Arial', 12, 'bold'), foreground='#34495E')
        self.style.configure('Agent.TFrame', background='#ECF0F1', relief='raised', borderwidth=2)
        self.style.configure('Input.TFrame', background='#FFFFFF', relief='sunken', borderwidth=1)
        self.style.configure('Results.TFrame', background='#F8F9FA', relief='raised', borderwidth=1)
    
    def create_menu(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Vessel", command=self.new_vessel)
        file_menu.add_command(label="Load Vessel Data", command=self.load_vessel_data)
        file_menu.add_command(label="Save Vessel Data", command=self.save_vessel_data)
        file_menu.add_separator()
        file_menu.add_command(label="Export Results", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Reset Fields", command=self.reset_fields)
        edit_menu.add_command(label="Load Template", command=self.load_template)
        
        # Analysis menu
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        analysis_menu.add_command(label="Run Surrogate Model", command=self.run_surrogate_model)
        analysis_menu.add_command(label="Run Multi-Objective Optimization", command=self.run_optimization)
        analysis_menu.add_command(label="Run Climate Analysis", command=self.run_climate_analysis)
        analysis_menu.add_separator()
        analysis_menu.add_command(label="Run Complete Analysis", command=self.run_complete_analysis)
        
        # Model menu
        model_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=model_menu)
        model_menu.add_command(label="Train Surrogate Model", command=self.train_surrogate_model)
        model_menu.add_command(label="Load Trained Model", command=self.load_trained_model)
        model_menu.add_separator()
        model_menu.add_command(label="Model Info", command=self.show_model_info)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_user_guide)
    
    def create_main_layout(self):
        """Create main application layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="SmartCAPEX AI", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(main_frame, 
                                  text="Intelligent Maritime Retrofit Decision Support",
                                  font=('Arial', 10, 'italic'), foreground='#7F8C8D')
        subtitle_label.pack(pady=(0, 20))
        
        # Create paned window for resizable panels
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Input
        left_frame = ttk.Frame(paned, width=300)
        paned.add(left_frame, weight=1)
        
        # Right panel - Results
        right_frame = ttk.Frame(paned, width=900)
        paned.add(right_frame, weight=2)
        
        self.left_frame = left_frame
        self.right_frame = right_frame
    
    def create_input_panel(self):
        """Create vessel input panel"""
        # Input section title
        input_title = ttk.Label(self.left_frame, text="Vessel Data Input", 
                               style='Section.TLabel')
        input_title.pack(pady=(0, 10))
        
        # Create notebook for different input methods
        input_notebook = ttk.Notebook(self.left_frame)
        input_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Manual input tab
        manual_frame = ttk.Frame(input_notebook)
        input_notebook.add(manual_frame, text="Manual Input")
        self.create_manual_input(manual_frame)
        
        # Template tab
        template_frame = ttk.Frame(input_notebook)
        input_notebook.add(template_frame, text="Templates")
        self.create_template_input(template_frame)
        
        # Quick preset tab
        preset_frame = ttk.Frame(input_notebook)
        input_notebook.add(preset_frame, text="Quick Presets")
        self.create_preset_input(preset_frame)
    
    def create_manual_input(self, parent):
        """Create manual input form"""
        # Create scrollable frame
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Input fields
        self.input_fields = {}
        
        # Basic vessel information
        self.add_section(scrollable_frame, "Basic Vessel Information")
        
        basic_fields = [
            ('dwt', 'DWT (tons)', 5000),
            ('length', 'Length (m)', 100),
            ('breadth', 'Breadth (m)', 16),
            ('draft', 'Draft (m)', 6.5),
            ('speed', 'Service Speed (knots)', 12),
            ('age', 'Age (years)', 15)
        ]
        
        for field_name, label, default in basic_fields:
            self.add_input_field(scrollable_frame, field_name, label, default)
        
        # Performance metrics
        self.add_section(scrollable_frame, "Performance Metrics")
        
        performance_fields = [
            ('fuel_consumption', 'Fuel Consumption (tons/day)', 18),
            ('co2_emission', 'CO2 Emissions (tons/day)', 55),
            ('cii_score', 'CII Score', 4.2),
            ('eedi_score', 'EEDI Score', 20)
        ]
        
        for field_name, label, default in performance_fields:
            self.add_input_field(scrollable_frame, field_name, label, default)
        
        # Environmental conditions
        self.add_section(scrollable_frame, "Environmental Conditions")
        
        env_fields = [
            ('wave_height', 'Wave Height (m)', 2.0),
            ('wind_speed', 'Wind Speed (knots)', 15),
            ('sea_state', 'Sea State (1-6)', 3)
        ]
        
        for field_name, label, default in env_fields:
            self.add_input_field(scrollable_frame, field_name, label, default)
        
        # Operational parameters
        self.add_section(scrollable_frame, "Operational Parameters")
        
        operational_fields = [
            ('load_factor', 'Load Factor (0-1)', 0.8),
            ('engine_efficiency', 'Engine Efficiency (0-1)', 0.42)
        ]
        
        for field_name, label, default in operational_fields:
            self.add_input_field(scrollable_frame, field_name, label, default)
        
        # Update button
        update_btn = ttk.Button(scrollable_frame, text="Update Analysis", 
                               command=self.update_vessel_data)
        update_btn.pack(pady=10)
    
    def add_section(self, parent, title):
        """Add a section header"""
        separator = ttk.Separator(parent, orient='horizontal')
        separator.pack(fill='x', pady=(10, 5))
        
        label = ttk.Label(parent, text=title, font=('Arial', 10, 'bold'))
        label.pack(anchor='w', pady=(0, 5))
    
    def add_input_field(self, parent, field_name, label, default):
        """Add an input field"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        label_widget = ttk.Label(frame, text=label, width=25)
        label_widget.pack(side='left', padx=(0, 5))
        
        entry = ttk.Entry(frame, width=15)
        entry.insert(0, str(default))
        entry.pack(side='left')
        
        self.input_fields[field_name] = entry
    
    def create_template_input(self, parent):
        """Create template selection interface"""
        label = ttk.Label(parent, text="Select Vessel Template:")
        label.pack(pady=10)
        
        # Template buttons
        templates = [
            ('koster_coaster', 'Koster Coaster (Typical)'),
            ('general_cargo', 'General Cargo Vessel'),
            ('bulk_carrier', 'Small Bulk Carrier')
        ]
        
        for template_id, template_name in templates:
            btn = ttk.Button(parent, text=template_name,
                           command=lambda t=template_id: self.load_template_data(t))
            btn.pack(pady=5, padx=20, fill='x')
    
    def create_preset_input(self, parent):
        """Create quick preset interface"""
        label = ttk.Label(parent, text="Quick Vessel Presets:")
        label.pack(pady=10)
        
        # Preset scenarios
        presets = [
            ('Old Koster (20+ years)', {'dwt': 4500, 'age': 22, 'fuel_consumption': 22, 'cii_score': 5.2}),
            ('Middle-aged Koster (10-15 years)', {'dwt': 5500, 'age': 12, 'fuel_consumption': 16, 'cii_score': 3.8}),
            ('Young Koster (<10 years)', {'dwt': 6000, 'age': 7, 'fuel_consumption': 14, 'cii_score': 3.2})
        ]
        
        for preset_name, preset_data in presets:
            btn = ttk.Button(parent, text=preset_name,
                           command=lambda p=preset_data: self.load_preset_data(p))
            btn.pack(pady=5, padx=20, fill='x')
    
    def create_results_panel(self):
        """Create results display panel"""
        # Results section title
        results_title = ttk.Label(self.right_frame, text="Analysis Results", 
                                 style='Section.TLabel')
        results_title.pack(pady=(0, 10))
        
        # Create notebook for different result types
        results_notebook = ttk.Notebook(self.right_frame)
        results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary results tab
        summary_frame = ttk.Frame(results_notebook)
        results_notebook.add(summary_frame, text="Summary")
        self.create_summary_display(summary_frame)
        
        # Charts tab
        charts_frame = ttk.Frame(results_notebook)
        results_notebook.add(charts_frame, text="Charts")
        self.create_charts_display(charts_frame)
        
        # Detailed results tab
        details_frame = ttk.Frame(results_notebook)
        results_notebook.add(details_frame, text="Details")
        self.create_details_display(details_frame)
        
        self.results_notebook = results_notebook
    
    def create_summary_display(self, parent):
        """Create summary results display"""
        # Text widget for summary
        self.summary_text = tk.Text(parent, height=20, width=80)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add default message
        self.summary_text.insert('1.0', "Run analysis to see results here...")
        self.summary_text.config(state=tk.DISABLED)
    
    def create_charts_display(self, parent):
        """Create charts display"""
        # Create matplotlib figure
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout()
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_details_display(self, parent):
        """Create detailed results display"""
        # Create treeview for detailed results
        columns = ('Parameter', 'Value', 'Unit')
        self.details_tree = ttk.Treeview(parent, columns=columns, show='headings')
        
        for col in columns:
            self.details_tree.heading(col, text=col)
            self.details_tree.column(col, width=150)
        
        self.details_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Label(self.root, text="Ready", relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_bar.config(text=message)
        self.root.update()
    
    def load_default_data(self):
        """Load default vessel data"""
        self.current_vessel_data = {
            'dwt': 5000,
            'length': 100,
            'breadth': 16,
            'draft': 6.5,
            'speed': 12,
            'age': 15,
            'fuel_consumption': 18,
            'co2_emission': 55,
            'cii_score': 4.2,
            'eedi_score': 20,
            'wave_height': 2.0,
            'wind_speed': 15,
            'sea_state': 3,
            'load_factor': 0.8,
            'engine_efficiency': 0.42,
            'fuel_type': 'HFO'
        }
        
        self.update_input_fields()
        self.update_status("Default vessel data loaded")
    
    def update_input_fields(self):
        """Update input fields with current vessel data"""
        for field_name, entry in self.input_fields.items():
            if field_name in self.current_vessel_data:
                entry.delete(0, tk.END)
                entry.insert(0, str(self.current_vessel_data[field_name]))
    
    def update_vessel_data(self):
        """Update vessel data from input fields"""
        for field_name, entry in self.input_fields.items():
            try:
                value = float(entry.get())
                self.current_vessel_data[field_name] = value
            except ValueError:
                messagebox.showwarning("Invalid Input", 
                                     f"Invalid value for {field_name}")
                return
        
        # Validate data
        is_valid, errors = self.asset_manager.validate_all_inputs(self.current_vessel_data)
        
        if not is_valid:
            error_msg = "\n".join([f"{field}: {error}" for field, error in errors.items()])
            messagebox.showerror("Validation Error", error_msg)
            return
        
        # Impute missing data
        self.current_vessel_data = self.asset_manager.impute_missing_data(
            self.current_vessel_data
        )
        
        self.update_input_fields()
        self.update_status("Vessel data updated and validated")
    
    def load_template_data(self, template_name):
        """Load data from template"""
        try:
            template_data = self.asset_manager.load_vessel_template(template_name)
            self.current_vessel_data.update(template_data)
            self.update_input_fields()
            self.update_status(f"Loaded template: {template_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load template: {str(e)}")
    
    def load_preset_data(self, preset_data):
        """Load preset data"""
        self.current_vessel_data.update(preset_data)
        self.update_input_fields()
        self.update_status("Preset data loaded")
    
    def train_surrogate_model(self):
        """Train the surrogate model"""
        self.update_status("Training surrogate model...")
        
        def train_worker():
            try:
                results = self.surrogate_modeler.train_models()
                self.is_model_trained = True
                self.update_status("Surrogate model trained successfully")
                
                # Show results (fix: use correct dictionary keys)
                messagebox.showinfo("Training Complete", 
                                  f"EANN Loss: {results['eann_loss']:.4f}\n"
                                  f"EANN MAE: {results['eann_mae']:.4f}\n"
                                  f"EANN RMSE: {results['eann_rmse']:.4f}\n"
                                  f"RF R² Score: {results['rf_score']:.4f}\n"
                                  f"GB R² Score: {results['gb_score']:.4f}")
            except Exception as e:
                self.update_status("Training failed")
                messagebox.showerror("Training Error", str(e))
        
        # Run training in separate thread
        threading.Thread(target=train_worker, daemon=True).start()
    
    def run_surrogate_model(self):
        """Run surrogate model prediction"""
        if not self.is_model_trained:
            messagebox.showwarning("Model Not Trained", 
                                 "Please train the surrogate model first.")
            return
        
        self.update_status("Running surrogate model...")
        
        try:
            # Run prediction
            results = self.surrogate_modeler.predict(self.current_vessel_data)
            
            # Update results display
            self.update_summary_display("Surrogate Model Results", results)
            self.update_details_display(results)
            
            self.update_status("Surrogate model analysis complete")
        except Exception as e:
            messagebox.showerror("Analysis Error", str(e))
    
    def run_optimization(self):
        """Run multi-objective optimization"""
        self.update_status("Running multi-objective optimization...")
        
        try:
            # Run optimization
            results = self.optimizer.optimize_scenarios(self.current_vessel_data)
            
            # Generate report
            report = self.optimizer.generate_report()
            
            # Update results display
            self.update_summary_display("Multi-Objective Optimization Results", report)
            self.update_optimization_charts(results)
            
            self.update_status("Optimization complete")
        except Exception as e:
            messagebox.showerror("Optimization Error", str(e))
    
    def run_climate_analysis(self):
        """Run climate impact analysis"""
        self.update_status("Running climate impact analysis...")
        
        try:
            # Run climate analysis
            risk_assessment = self.climate_guardian.calculate_climate_risk_assessment(
                self.current_vessel_data
            )
            
            # Update results display
            self.update_summary_display("Climate Impact Analysis", risk_assessment)
            self.update_climate_charts(risk_assessment)
            
            self.update_status("Climate analysis complete")
        except Exception as e:
            messagebox.showerror("Climate Analysis Error", str(e))
    
    def run_complete_analysis(self):
        """Run complete analysis with all agents"""
        self.update_status("Running complete analysis...")
        
        def analysis_worker():
            try:
                # Step 1: Surrogate Model
                if self.is_model_trained:
                    surrogate_results = self.surrogate_modeler.predict(self.current_vessel_data)
                else:
                    surrogate_results = {"status": "Model not trained"}
                
                # Step 2: Multi-Objective Optimization
                optimization_results = self.optimizer.optimize_scenarios(self.current_vessel_data)
                optimization_report = self.optimizer.generate_report()
                
                # Step 3: Climate Analysis
                climate_results = self.climate_guardian.calculate_climate_risk_assessment(
                    self.current_vessel_data
                )
                
                # Compile results
                self.analysis_results = {
                    'surrogate': surrogate_results,
                    'optimization': optimization_report,
                    'climate': climate_results
                }
                
                # Update display
                self.update_complete_analysis_display()
                
                self.update_status("Complete analysis finished")
                
            except Exception as e:
                self.update_status("Analysis failed")
                messagebox.showerror("Analysis Error", str(e))
        
        # Run analysis in separate thread
        threading.Thread(target=analysis_worker, daemon=True).start()
    
    def update_summary_display(self, title, results):
        """Update summary display"""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete('1.0', tk.END)
        
        self.summary_text.insert('1.0', f"=== {title} ===\n\n")
        
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    self.summary_text.insert(tk.END, f"{key}: {value:.3f}\n")
                else:
                    self.summary_text.insert(tk.END, f"{key}: {value}\n")
        else:
            self.summary_text.insert(tk.END, str(results))
        
        self.summary_text.config(state=tk.DISABLED)
    
    def update_details_display(self, results):
        """Update details display"""
        # Clear existing items
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)
        
        # Add new items
        if isinstance(results, dict):
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    self.details_tree.insert('', 'end', values=(key, f"{value:.3f}", ""))
                else:
                    self.details_tree.insert('', 'end', values=(key, str(value), ""))
    
    def update_optimization_charts(self, results):
        """Update optimization charts"""
        # Clear previous charts
        for ax in self.axes.flat:
            ax.clear()
        
        # Get scenario names (exclude non-scenario keys)
        scenarios = [k for k in results.keys() if k in ['current', 'retrofit', 'newbuild']]
        npv_values = [results[s]['npv'] for s in scenarios]
        
        self.axes[0, 0].bar(scenarios, npv_values, color=['#E74C3C', '#3498DB', '#2ECC71'])
        self.axes[0, 0].set_title('NPV Comparison')
        self.axes[0, 0].set_ylabel('NPV (USD)')
        
        # Chart 2: Environmental Scores
        env_scores = [results[s]['environmental_score'] for s in scenarios]
        
        self.axes[0, 1].bar(scenarios, env_scores, color=['#E74C3C', '#3498DB', '#2ECC71'])
        self.axes[0, 1].set_title('Environmental Scores')
        self.axes[0, 1].set_ylabel('Score (0-100)')
        
        # Chart 3: TOPSIS Scores (was 'mcdm_score', now 'topsis_score')
        topsis_scores = [results[s].get('topsis_score', 0) for s in scenarios]
        
        self.axes[1, 0].bar(scenarios, topsis_scores, color=['#E74C3C', '#3498DB', '#2ECC71'])
        self.axes[1, 0].set_title('TOPSIS Scores')
        self.axes[1, 0].set_ylabel('Score (0-1)')
        
        # Chart 4: Pareto Front
        pareto_scenarios = self.optimizer.pareto_front
        if pareto_scenarios:
            pareto_indices = [scenarios.index(s) for s in pareto_scenarios if s in scenarios]
            pareto_npv = [npv_values[i] for i in pareto_indices]
            pareto_env = [env_scores[i] for i in pareto_indices]
            
            self.axes[1, 1].scatter(npv_values, env_scores, c='lightblue', s=100, alpha=0.6)
            self.axes[1, 1].scatter(pareto_npv, pareto_env, c='red', s=150, marker='*')
            self.axes[1, 1].set_xlabel('NPV (USD)')
            self.axes[1, 1].set_ylabel('Environmental Score')
            self.axes[1, 1].set_title('Pareto Front')
        
        self.canvas.draw()
    
    def update_climate_charts(self, risk_assessment):
        """Update climate analysis charts"""
        # Clear previous charts
        for ax in self.axes.flat:
            ax.clear()
        
        temporal_analysis = risk_assessment['temporal_analysis']
        years = list(temporal_analysis['yearly_projections'].keys())
        
        # Chart 1: Additional Costs Over Time
        costs = [temporal_analysis['yearly_projections'][year]['annual_additional_costs']['total_additional_cost'] 
                for year in years]
        
        self.axes[0, 0].plot(years, costs, marker='o')
        self.axes[0, 0].set_title('Additional Costs Over Time')
        self.axes[0, 0].set_xlabel('Year')
        self.axes[0, 0].set_ylabel('Additional Cost (USD/year)')
        
        # Chart 2: Wave Height Projection
        wave_heights = [temporal_analysis['yearly_projections'][year]['environmental_conditions']['wave_height']['mean'] 
                       for year in years]
        
        self.axes[0, 1].plot(years, wave_heights, marker='s', color='blue')
        self.axes[0, 1].set_title('Wave Height Projection')
        self.axes[0, 1].set_xlabel('Year')
        self.axes[0, 1].set_ylabel('Wave Height (m)')
        
        # Chart 3: Risk Components
        risk_components = ['Physical', 'Transition', 'Regulatory']
        risk_values = [risk_assessment['physical_risk'], 
                      risk_assessment['transition_risk'], 
                      risk_assessment['regulatory_risk']]
        
        self.axes[1, 0].bar(risk_components, risk_values, color=['red', 'orange', 'yellow'])
        self.axes[1, 0].set_title('Risk Components')
        self.axes[1, 0].set_ylabel('Risk Level (0-1)')
        
        # Chart 4: Resistance Penalty Over Time
        resistance_penalties = [temporal_analysis['yearly_projections'][year]['resistance_penalty'] 
                               for year in years]
        
        self.axes[1, 1].plot(years, resistance_penalties, marker='^', color='green')
        self.axes[1, 1].set_title('Resistance Penalty Over Time')
        self.axes[1, 1].set_xlabel('Year')
        self.axes[1, 1].set_ylabel('Resistance Penalty Factor')
        
        self.canvas.draw()
    
    def update_complete_analysis_display(self):
        """Update display with complete analysis results"""
        self.update_summary_display("Complete Analysis Results", self.analysis_results)
        
        # Update charts based on active tab
        if 'optimization' in self.analysis_results:
            self.update_optimization_charts(self.analysis_results['optimization']['scenarios'])
    
    def new_vessel(self):
        """Create new vessel configuration"""
        self.load_default_data()
        self.analysis_results = {}
        self.clear_displays()
        self.update_status("New vessel configuration created")
    
    def load_vessel_data(self):
        """Load vessel data from file"""
        filepath = filedialog.askopenfilename(
            title="Load Vessel Data",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                data = self.asset_manager.import_data(filepath)
                self.current_vessel_data.update(data)
                self.update_input_fields()
                self.update_status(f"Loaded vessel data from {filepath}")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))
    
    def save_vessel_data(self):
        """Save vessel data to file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Vessel Data",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.asset_manager.export_data(self.current_vessel_data, filepath)
                self.update_status(f"Saved vessel data to {filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e))
    
    def export_results(self):
        """Export analysis results"""
        if not self.analysis_results:
            messagebox.showwarning("No Results", "No analysis results to export")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Export Analysis Results",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.asset_manager.export_data(self.analysis_results, filepath)
                self.update_status(f"Exported results to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
    
    def reset_fields(self):
        """Reset all input fields"""
        for entry in self.input_fields.values():
            entry.delete(0, tk.END)
        
        self.current_vessel_data = {}
        self.update_status("Fields reset")
    
    def load_template(self):
        """Show template selection dialog"""
        # This would show a dialog, but for now just use the existing template interface
        self.update_status("Use the Templates tab to select a vessel template")
    
    def load_trained_model(self):
        """Load a pre-trained surrogate model"""
        messagebox.showinfo("Model Loading", "Model loading functionality would be implemented here")
    
    def show_model_info(self):
        """Show model information"""
        info_text = """
SmartCAPEX AI Model Information:

Agents:
1. Surrogate Modeler (EANN) - Physics-informed neural network
2. Multi-Objective Optimizer - Pareto-based optimization
3. Climate Guardian - Temporal projection analysis
4. Asset Manager - Data validation and management

Literature Base:
- AI-Based Decision Support Systems for Retrofit (Bocaneala et al.)
- Surrogate Modeling techniques (Westermann et al., 2020)
- Emotional ANN (Aljahdali et al., 2025)
- Pareto Optimality (Rosso et al., 2020)
        """
        messagebox.showinfo("Model Information", info_text)
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
SmartCAPEX AI v1.0

Intelligent Multi-Agent Architecture for Maritime Retrofit Decisions

Based on AI-Based Decision Support Systems methodology for retrofit projects.

Authors: AI Assistant Implementation
Literature: Bocaneala et al., Westermann et al., Aljahdali et al., Rosso et al.

© 2025 SmartCAPEX AI
        """
        messagebox.showinfo("About SmartCAPEX AI", about_text)
    
    def show_user_guide(self):
        """Show user guide"""
        guide_text = """
SmartCAPEX AI User Guide:

1. Input Vessel Data:
   - Use Manual Input tab for custom data
   - Use Templates tab for predefined vessel types
   - Use Quick Presets for common scenarios

2. Train Models:
   - Go to Model > Train Surrogate Model
   - Wait for training to complete

3. Run Analysis:
   - Use Analysis menu to run individual agents
   - Or use "Run Complete Analysis" for all agents

4. View Results:
   - Summary tab shows key results
   - Charts tab displays visualizations
   - Details tab shows complete data

5. Export Results:
   - Use File > Export Results to save analysis
        """
        messagebox.showinfo("User Guide", guide_text)
    
    def clear_displays(self):
        """Clear all result displays"""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete('1.0', tk.END)
        self.summary_text.insert('1.0', "Run analysis to see results here...")
        self.summary_text.config(state=tk.DISABLED)
        
        # Clear charts
        for ax in self.axes.flat:
            ax.clear()
        self.canvas.draw()
        
        # Clear details
        for item in self.details_tree.get_children():
            self.details_tree.delete(item)


def main():
    """Main entry point for the GUI application"""
    root = tk.Tk()
    app = SmartCAPEXMainWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
