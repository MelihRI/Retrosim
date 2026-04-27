"""
Retrosim - PDF Report Generator
====================================

Generates professional PDF reports containing:
- Vessel specifications
- Financial analysis (NPV, ROI)
- Scenario comparisons (Do-Nothing, Retrofit, New Build)
- TOPSIS/IPSO optimization results
- Climate projections
- Visualizations and charts
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

# PDF Generation using Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches


class RetrosimReportGenerator:
    """
    Professional PDF Report Generator for Retrosim
    
    Features:
    - Multi-page PDF with cover page
    - Vessel specifications table
    - Financial analysis charts
    - Scenario comparison graphs
    - Optimization results
    - Climate impact analysis
    """
    
    def __init__(self):
        self.company_name = "Retrosim"
        self.company_tagline = "Intelligent Maritime Investment Decision Support"
        
        # Color scheme (matching app theme)
        self.colors = {
            'primary': '#1f6feb',
            'secondary': '#58a6ff',
            'success': '#238636',
            'warning': '#d29922',
            'danger': '#da3633',
            'bg_dark': '#0d1117',
            'bg_light': '#161b22',
            'text': '#c9d1d9',
            'border': '#30363d'
        }
        
        # Configure matplotlib for dark theme
        plt.style.use('dark_background')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.facecolor'] = self.colors['bg_dark']
        plt.rcParams['figure.facecolor'] = self.colors['bg_dark']
        plt.rcParams['text.color'] = self.colors['text']
        plt.rcParams['axes.labelcolor'] = self.colors['text']
        plt.rcParams['xtick.color'] = self.colors['text']
        plt.rcParams['ytick.color'] = self.colors['text']
    
    def generate_report(self, 
                       vessel_data: Dict[str, Any],
                       analysis_results: Dict[str, Any] = None,
                       output_path: str = None) -> str:
        """
        Generate a comprehensive PDF report.
        
        Args:
            vessel_data: Dictionary containing vessel specifications
            analysis_results: Dictionary containing analysis outputs
            output_path: Path to save the PDF (auto-generated if None)
            
        Returns:
            Path to the generated PDF file
        """
        if output_path is None:
            vessel_name = vessel_data.get('name', 'Vessel').replace('/', '_').replace(' ', '_')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"Retrosim_Report_{vessel_name}_{timestamp}.pdf"
        
        # Ensure .pdf extension
        if not output_path.endswith('.pdf'):
            output_path += '.pdf'
        
        # Create PDF
        with PdfPages(output_path) as pdf:
            # Page 1: Cover
            self._create_cover_page(pdf, vessel_data)
            
            # Page 2: Vessel Specifications
            self._create_vessel_specs_page(pdf, vessel_data)
            
            # Page 3: Financial Summary
            if analysis_results:
                self._create_financial_page(pdf, vessel_data, analysis_results)
            
            # Page 4: Scenario Comparison
            if analysis_results and 'scenarios' in analysis_results:
                self._create_scenario_page(pdf, analysis_results)
            
            # Page 5: Optimization Results
            if analysis_results and ('topsis' in analysis_results or 'ipso' in analysis_results):
                self._create_optimization_page(pdf, analysis_results)
            
            # Page 6: Climate Analysis
            if analysis_results and 'climate' in analysis_results:
                self._create_climate_page(pdf, analysis_results)
            
            # Final Page: Summary & Recommendations
            self._create_summary_page(pdf, vessel_data, analysis_results)
        
        plt.close('all')
        return output_path
    
    def _create_cover_page(self, pdf: PdfPages, vessel_data: Dict):
        """Create professional cover page"""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 size
        ax.axis('off')
        
        # Background gradient effect (simulated with rectangle)
        bg_rect = FancyBboxPatch((0, 0), 1, 1, 
                                  boxstyle="round,pad=0",
                                  facecolor=self.colors['bg_dark'],
                                  transform=ax.transAxes)
        ax.add_patch(bg_rect)
        
        # Company Logo/Title
        ax.text(0.5, 0.85, "🚢 Retrosim", 
                fontsize=32, fontweight='bold', 
                ha='center', va='center',
                color=self.colors['secondary'],
                transform=ax.transAxes)
        
        ax.text(0.5, 0.78, self.company_tagline, 
                fontsize=12, ha='center', va='center',
                color=self.colors['text'],
                style='italic',
                transform=ax.transAxes)
        
        # Decorative line
        ax.axhline(y=0.72, xmin=0.2, xmax=0.8, 
                   color=self.colors['primary'], linewidth=2)
        
        # Report Title
        ax.text(0.5, 0.60, "YATIRIM ANALİZ RAPORU", 
                fontsize=24, fontweight='bold', 
                ha='center', va='center',
                color='white',
                transform=ax.transAxes)
        
        ax.text(0.5, 0.52, "Investment Analysis Report", 
                fontsize=14, ha='center', va='center',
                color=self.colors['text'],
                transform=ax.transAxes)
        
        # Vessel Info Box
        vessel_name = vessel_data.get('name', 'Unknown Vessel')
        vessel_type = vessel_data.get('type', 'Unknown Type')
        dwt = vessel_data.get('dwt', 0)
        
        box_y = 0.38
        ax.text(0.5, box_y, f"─────────────────────────", 
                fontsize=12, ha='center', color=self.colors['border'],
                transform=ax.transAxes)
        
        ax.text(0.5, box_y - 0.05, f"📋 {vessel_name}", 
                fontsize=18, fontweight='bold', ha='center',
                color=self.colors['secondary'],
                transform=ax.transAxes)
        
        ax.text(0.5, box_y - 0.10, f"{vessel_type} | {dwt:,} DWT", 
                fontsize=12, ha='center',
                color=self.colors['text'],
                transform=ax.transAxes)
        
        ax.text(0.5, box_y - 0.15, f"─────────────────────────", 
                fontsize=12, ha='center', color=self.colors['border'],
                transform=ax.transAxes)
        
        # Date and metadata
        current_date = datetime.now().strftime('%d %B %Y')
        ax.text(0.5, 0.12, f"Rapor Tarihi: {current_date}", 
                fontsize=10, ha='center',
                color=self.colors['text'],
                transform=ax.transAxes)
        
        ax.text(0.5, 0.08, "Gizli - Sadece Dahili Kullanım İçin", 
                fontsize=9, ha='center',
                color=self.colors['warning'],
                style='italic',
                transform=ax.transAxes)
        
        # Footer
        ax.text(0.5, 0.02, "Powered by Retrosim v2.0 | Multi-Agent Decision Support System", 
                fontsize=8, ha='center',
                color='#666',
                transform=ax.transAxes)
        
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_vessel_specs_page(self, pdf: PdfPages, vessel_data: Dict):
        """Create vessel specifications page with tables"""
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.suptitle('📊 Gemi Teknik Özellikleri\nVessel Technical Specifications', 
                     fontsize=16, fontweight='bold', color=self.colors['secondary'])
        
        # Hide all axes initially
        for ax_row in axes:
            for ax in ax_row:
                ax.axis('off')
        
        # Table 1: Main Dimensions
        ax1 = axes[0, 0]
        ax1.set_title('📐 Ana Boyutlar', fontsize=12, fontweight='bold', 
                      color=self.colors['secondary'], loc='left')
        
        dims_data = [
            ['LOA (Boy)', f"{vessel_data.get('loa', 0):.1f} m"],
            ['Beam (Genişlik)', f"{vessel_data.get('beam', 0):.1f} m"],
            ['Draft (Su Çekimi)', f"{vessel_data.get('draft', 0):.1f} m"],
            ['Depth (Derinlik)', f"{vessel_data.get('depth', 0):.1f} m"],
            ['DWT', f"{vessel_data.get('dwt', 0):,} ton"],
            ['Yaş', f"{vessel_data.get('age', 0)} yıl"],
        ]
        
        table1 = ax1.table(cellText=dims_data, 
                          colLabels=['Parametre', 'Değer'],
                          loc='center',
                          cellLoc='left')
        table1.auto_set_font_size(False)
        table1.set_fontsize(9)
        table1.scale(1.2, 1.8)
        self._style_table(table1)
        
        # Table 2: Form Coefficients
        ax2 = axes[0, 1]
        ax2.set_title('📈 Form Katsayıları', fontsize=12, fontweight='bold',
                      color=self.colors['secondary'], loc='left')
        
        coef_data = [
            ['Blok Katsayısı (Cb)', f"{vessel_data.get('cb', 0):.3f}"],
            ['Prizmatik (Cp)', f"{vessel_data.get('cp', 0):.3f}"],
            ['Midship (Cm)', f"{vessel_data.get('cm', 0):.3f}"],
            ['Bulb Boyu', f"{vessel_data.get('bulb_length', 0):.1f} m"],
            ['Pervane Çapı', f"{vessel_data.get('prop_dia', 0):.1f} m"],
        ]
        
        table2 = ax2.table(cellText=coef_data,
                          colLabels=['Parametre', 'Değer'],
                          loc='center',
                          cellLoc='left')
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1.2, 1.8)
        self._style_table(table2)
        
        # Table 3: Performance
        ax3 = axes[1, 0]
        ax3.set_title('⚡ Performans', fontsize=12, fontweight='bold',
                      color=self.colors['secondary'], loc='left')
        
        perf_data = [
            ['Servis Hızı', f"{vessel_data.get('speed', 0):.1f} knot"],
            ['Motor Gücü', f"{vessel_data.get('engine_power', 0):,} kW"],
            ['SFOC', f"{vessel_data.get('sfoc', 0):.1f} g/kWh"],
            ['CII Derecesi', f"{vessel_data.get('cii', 'N/A')}"],
            ['EEDI', f"{vessel_data.get('eedi', 0):.2f}"],
        ]
        
        table3 = ax3.table(cellText=perf_data,
                          colLabels=['Parametre', 'Değer'],
                          loc='center',
                          cellLoc='left')
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1.2, 1.8)
        self._style_table(table3)
        
        # Table 4: Economics
        ax4 = axes[1, 1]
        ax4.set_title('💰 Ekonomik Veriler', fontsize=12, fontweight='bold',
                      color=self.colors['secondary'], loc='left')
        
        econ_data = [
            ['Gemi Değeri', f"${vessel_data.get('value', 0):,}"],
            ['Yıllık OPEX', f"${vessel_data.get('opex', 0):,}/gün"],
            ['Tahmini Yakıt/Gün', f"~{vessel_data.get('engine_power', 0) * vessel_data.get('sfoc', 0) * 24 / 1e6:.1f} ton"],
        ]
        
        table4 = ax4.table(cellText=econ_data,
                          colLabels=['Parametre', 'Değer'],
                          loc='center',
                          cellLoc='left')
        table4.auto_set_font_size(False)
        table4.set_fontsize(9)
        table4.scale(1.2, 1.8)
        self._style_table(table4)
        
        plt.tight_layout()
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_financial_page(self, pdf: PdfPages, vessel_data: Dict, analysis_results: Dict):
        """Create financial analysis page with charts"""
        fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
        fig.suptitle('💰 Finansal Analiz\nFinancial Analysis', 
                     fontsize=16, fontweight='bold', color=self.colors['secondary'])
        
        # Chart 1: NPV Comparison (Bar Chart)
        ax1 = axes[0, 0]
        scenarios = ['Do-Nothing', 'Retrofit', 'New Build']
        npv_values = analysis_results.get('npv_comparison', [-500000, 1200000, 2500000])
        colors = [self.colors['danger'], self.colors['warning'], self.colors['success']]
        
        bars = ax1.bar(scenarios, [v/1e6 for v in npv_values], color=colors, edgecolor='white')
        ax1.set_title('NPV Karşılaştırması (20 Yıl)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('NPV (Million $)')
        ax1.axhline(y=0, color='white', linestyle='-', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, npv_values):
            height = bar.get_height()
            ax1.annotate(f'${val/1e6:.1f}M',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -10),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, color='white')
        
        # Chart 2: Cash Flow (Line Chart)
        ax2 = axes[0, 1]
        years = list(range(0, 21))
        cash_flow = analysis_results.get('cash_flow', [v * (0.9 + 0.1*np.random.random()) for v in np.linspace(-1000000, 2000000, 21)])
        
        ax2.plot(years, [v/1e6 for v in cash_flow], color=self.colors['secondary'], 
                 linewidth=2, marker='o', markersize=4)
        ax2.fill_between(years, [v/1e6 for v in cash_flow], 0, 
                         where=[v >= 0 for v in cash_flow],
                         alpha=0.3, color=self.colors['success'])
        ax2.fill_between(years, [v/1e6 for v in cash_flow], 0, 
                         where=[v < 0 for v in cash_flow],
                         alpha=0.3, color=self.colors['danger'])
        ax2.set_title('Kümülatif Nakit Akışı', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Yıl')
        ax2.set_ylabel('Nakit Akışı (Million $)')
        ax2.axhline(y=0, color='white', linestyle='--', linewidth=0.5)
        ax2.grid(True, alpha=0.2)
        
        # Chart 3: ROI Breakdown (Pie Chart)
        ax3 = axes[1, 0]
        roi_labels = ['Yakıt Tasarrufu', 'Karbon Vergisi\nKaçınma', 'OPEX\nAzalması', 'Navlun\nGeliri']
        roi_values = analysis_results.get('roi_breakdown', [45, 25, 15, 15])
        explode = (0.05, 0, 0, 0)
        
        wedges, texts, autotexts = ax3.pie(roi_values, labels=roi_labels, autopct='%1.1f%%',
                                           colors=[self.colors['success'], self.colors['secondary'],
                                                   self.colors['warning'], '#9966ff'],
                                           explode=explode, startangle=90)
        ax3.set_title('ROI Bileşenleri', fontsize=11, fontweight='bold')
        
        # Chart 4: Key Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        metrics = [
            ('📈 Toplam NPV', f"${analysis_results.get('total_npv', 2500000)/1e6:.2f}M"),
            ('⏱️ Geri Dönüş Süresi', f"{analysis_results.get('payback_years', 4.5):.1f} Yıl"),
            ('📊 İç Verim Oranı (IRR)', f"{analysis_results.get('irr', 18.5):.1f}%"),
            ('💰 Yatırım Değeri', f"${analysis_results.get('investment', 850000)/1e6:.2f}M"),
            ('🌍 CO2 Azaltımı', f"{analysis_results.get('co2_reduction', 25):.0f}%"),
        ]
        
        ax4.text(0.5, 0.95, '📊 Özet Metrikler', fontsize=14, fontweight='bold',
                 ha='center', transform=ax4.transAxes, color=self.colors['secondary'])
        
        for i, (label, value) in enumerate(metrics):
            y_pos = 0.80 - i * 0.15
            ax4.text(0.1, y_pos, label, fontsize=11, transform=ax4.transAxes)
            ax4.text(0.9, y_pos, value, fontsize=11, fontweight='bold',
                     ha='right', transform=ax4.transAxes, color=self.colors['success'])
        
        plt.tight_layout()
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_scenario_page(self, pdf: PdfPages, analysis_results: Dict):
        """Create scenario comparison page"""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        ax.text(0.5, 0.95, '🔄 Senaryo Karşılaştırması\nScenario Comparison', 
                fontsize=16, fontweight='bold', ha='center',
                color=self.colors['secondary'], transform=ax.transAxes)
        
        scenarios_info = analysis_results.get('scenarios', {
            'do_nothing': {'npv': -500000, 'risk': 'Yüksek', 'recommendation': 'Tavsiye Edilmez'},
            'retrofit': {'npv': 1200000, 'risk': 'Orta', 'recommendation': 'Tavsiye Edilir'},
            'new_build': {'npv': 2500000, 'risk': 'Düşük', 'recommendation': 'En İyi Seçenek'}
        })
        
        # Info text
        ax.text(0.5, 0.75, 
                "Retrosim, geminiz için 3 farklı senaryo değerlendirmiştir:\n\n"
                "1. DO-NOTHING: Mevcut durumda kalma\n"
                "2. RETROFIT: Enerji verimliliği iyileştirmeleri\n"
                "3. NEW BUILD: Yeni gemi inşası\n\n"
                "Detaylı analiz sonuçları aşağıda verilmiştir.",
                fontsize=11, ha='center', va='top',
                color=self.colors['text'], transform=ax.transAxes,
                wrap=True)
        
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_optimization_page(self, pdf: PdfPages, analysis_results: Dict):
        """Create TOPSIS/IPSO optimization results page"""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        ax.text(0.5, 0.95, '📊 Optimizasyon Sonuçları\nOptimization Results', 
                fontsize=16, fontweight='bold', ha='center',
                color=self.colors['secondary'], transform=ax.transAxes)
        
        # TOPSIS Results
        if 'topsis' in analysis_results:
            ax.text(0.5, 0.85, 'TOPSIS - Çok Kriterli Karar Analizi', 
                    fontsize=14, fontweight='bold', ha='center',
                    color=self.colors['success'], transform=ax.transAxes)
        
        # IPSO Results  
        if 'ipso' in analysis_results:
            ax.text(0.5, 0.55, 'IPSO - Pareto Optimizasyonu', 
                    fontsize=14, fontweight='bold', ha='center',
                    color=self.colors['warning'], transform=ax.transAxes)
        
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_climate_page(self, pdf: PdfPages, analysis_results: Dict):
        """Create climate impact analysis page"""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        ax.text(0.5, 0.95, '🌍 İklim Etki Analizi\nClimate Impact Analysis', 
                fontsize=16, fontweight='bold', ha='center',
                color=self.colors['secondary'], transform=ax.transAxes)
        
        climate = analysis_results.get('climate', {})
        
        ax.text(0.5, 0.80, 
                f"Hedef Yıl: {climate.get('target_year', 2030)}\n"
                f"Senaryo: {climate.get('scenario', 'RCP 4.5')}\n"
                f"Risk Skoru: {climate.get('risk_score', 45)}%",
                fontsize=12, ha='center', va='top',
                color=self.colors['text'], transform=ax.transAxes)
        
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _create_summary_page(self, pdf: PdfPages, vessel_data: Dict, analysis_results: Dict):
        """Create summary and recommendations page"""
        fig, ax = plt.subplots(figsize=(8.27, 11.69))
        ax.axis('off')
        
        ax.text(0.5, 0.95, '✅ Özet ve Tavsiyeler\nSummary & Recommendations', 
                fontsize=16, fontweight='bold', ha='center',
                color=self.colors['secondary'], transform=ax.transAxes)
        
        # Recommendation Box
        ax.text(0.5, 0.80, '─'*50, fontsize=10, ha='center', 
                color=self.colors['border'], transform=ax.transAxes)
        
        recommendation = analysis_results.get('recommendation', 'RETROFIT') if analysis_results else 'RETROFIT'
        ax.text(0.5, 0.72, f'🎯 ÖNERİLEN STRATEJİ: {recommendation}', 
                fontsize=18, fontweight='bold', ha='center',
                color=self.colors['success'], transform=ax.transAxes)
        
        ax.text(0.5, 0.65, '─'*50, fontsize=10, ha='center', 
                color=self.colors['border'], transform=ax.transAxes)
        
        # Key takeaways
        takeaways = [
            "• Retrofit uygulaması ile yakıt tüketiminde %15-25 tasarruf sağlanabilir",
            "• Karbon vergisi maliyetleri önemli ölçüde azaltılabilir",
            "• CII derecelendirmesi iyileştirilebilir (C → B veya A)",
            "• Yatırım geri dönüş süresi 3-5 yıl arasındadır",
            "• 20 yıllık NPV pozitif ve cazip yatırım getirisi sunar"
        ]
        
        ax.text(0.1, 0.55, 'Temel Bulgular:', fontsize=12, fontweight='bold',
                color=self.colors['secondary'], transform=ax.transAxes)
        
        for i, takeaway in enumerate(takeaways):
            ax.text(0.1, 0.48 - i*0.05, takeaway, fontsize=10,
                    color=self.colors['text'], transform=ax.transAxes)
        
        # Disclaimer
        ax.text(0.5, 0.10, 
                "Bu rapor Retrosim tarafından otomatik olarak oluşturulmuştur.\n"
                "Yatırım kararları vermeden önce uzman görüşü alınması tavsiye edilir.",
                fontsize=8, ha='center', style='italic',
                color='#666', transform=ax.transAxes)
        
        ax.text(0.5, 0.03, f"© {datetime.now().year} Retrosim - Tüm Hakları Saklıdır", 
                fontsize=8, ha='center', color='#444', transform=ax.transAxes)
        
        pdf.savefig(fig, facecolor=self.colors['bg_dark'])
        plt.close(fig)
    
    def _style_table(self, table):
        """Apply dark theme styling to matplotlib table"""
        for key, cell in table.get_celld().items():
            cell.set_edgecolor(self.colors['border'])
            cell.set_facecolor(self.colors['bg_dark'])
            cell.set_text_props(color=self.colors['text'])
            
            # Header row
            if key[0] == 0:
                cell.set_facecolor(self.colors['primary'])
                cell.set_text_props(color='white', fontweight='bold')


# Quick test function
def generate_sample_report():
    """Generate a sample report for testing"""
    generator = RetrosimReportGenerator()
    
    sample_vessel = {
        'name': 'M/V Test Vessel',
        'type': 'Bulk Carrier',
        'dwt': 55000,
        'loa': 190.0,
        'beam': 32.2,
        'draft': 12.5,
        'depth': 18.0,
        'cb': 0.82,
        'cp': 0.84,
        'cm': 0.98,
        'bulb_length': 5.5,
        'prop_dia': 6.5,
        'speed': 12.5,
        'engine_power': 8500,
        'sfoc': 175.0,
        'cii': 'C',
        'eedi': 12.5,
        'value': 15000000,
        'opex': 4500,
        'age': 10
    }
    
    sample_analysis = {
        'npv_comparison': [-500000, 1200000, 2500000],
        'cash_flow': list(np.cumsum(np.random.randn(21) * 200000 + 150000)),
        'roi_breakdown': [45, 25, 15, 15],
        'total_npv': 2500000,
        'payback_years': 4.5,
        'irr': 18.5,
        'investment': 850000,
        'co2_reduction': 25,
        'recommendation': 'RETROFIT'
    }
    
    output = generator.generate_report(sample_vessel, sample_analysis)
    print(f"✅ Sample report generated: {output}")
    return output


if __name__ == "__main__":
    generate_sample_report()
