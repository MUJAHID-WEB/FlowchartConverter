import os
import sys
import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import subprocess
import zipfile
import tempfile
import re
import json
import time
import requests
import threading
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import base64
import io
import math
from pathlib import Path
import webbrowser
import shutil
import ollama
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Set Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class AdvancedOllamaManager:
    """Enhanced Ollama LLM manager with Mistral and accuracy tracking"""
    
    def __init__(self, model_name="mistral"):
        self.model_name = model_name
        self.ollama_running = False
        self.performance_metrics = []
        self.available_models = self.get_available_models()
        self.check_and_start_ollama()
        
    def get_available_models(self):
        """Get list of available models"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                models = response.json()
                return [model['name'] for model in models.get('models', [])]
        except:
            pass
        return []
    
    def check_and_start_ollama(self):
        """Check if Ollama is running and start if needed"""
        try:
            print("🔄 Checking Ollama status...")
            
            # Check if Ollama is already running
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    self.ollama_running = True
                    print("✅ Ollama is already running")
                    # Update available models
                    self.available_models = self.get_available_models()
                    print(f"📋 Available models: {', '.join(self.available_models)}")
                    return
            except:
                pass
            
            # Try to start Ollama
            print("Starting Ollama...")
            
            # Check if Ollama is installed
            try:
                result = subprocess.run(["ollama", "--version"], 
                                      capture_output=True, 
                                      text=True,
                                      creationflags=subprocess.CREATE_NO_WINDOW)
                if result.returncode != 0:
                    print("⚠️ Ollama not found. Please install Ollama from https://ollama.com/")
                    return
            except FileNotFoundError:
                print("⚠️ Ollama not found. Please install Ollama from https://ollama.com/")
                return
            
            # Start Ollama in background
            print("🔄 Starting Ollama server in background...")
            try:
                self.ollama_process = subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
                
                # Wait for server to start
                for _ in range(15):
                    try:
                        response = requests.get("http://localhost:11434/api/tags", timeout=2)
                        if response.status_code == 200:
                            break
                    except:
                        time.sleep(2)
                
                # Update available models
                self.available_models = self.get_available_models()
                print(f"📋 Available models: {', '.join(self.available_models)}")
                
                # Check if our model exists
                if self.model_name not in self.available_models:
                    print(f"⚠️ {self.model_name} not found. Available models: {', '.join(self.available_models)}")
                    # Try to use available model
                    if self.available_models:
                        self.model_name = self.available_models[0].split(':')[0]
                        print(f"📦 Using available model: {self.model_name}")
                    else:
                        print("❌ No models available. Please pull a model first.")
                        print("💡 Try: ollama pull llama2 or ollama pull mistral")
                        self.ollama_running = False
                        return
                
                self.ollama_running = True
                print(f"✅ Ollama server started with {self.model_name} model")
                
            except Exception as e:
                print(f"⚠️ Could not start Ollama: {e}")
                print("Please start Ollama manually and ensure it's running on http://localhost:11434")
                
        except Exception as e:
            print(f"⚠️ Error checking Ollama: {e}")
    
    def calculate_text_accuracy(self, original_text, extracted_text):
        """Calculate accuracy of text extraction"""
        if not original_text or not extracted_text:
            return 0.0
        
        # Simple word-based accuracy
        original_words = set(original_text.lower().split())
        extracted_words = set(extracted_text.lower().split())
        
        if not original_words:
            return 0.0
        
        # Calculate intersection
        common_words = original_words.intersection(extracted_words)
        accuracy = len(common_words) / len(original_words)
        
        return round(accuracy * 100, 2)
    
    def calculate_structure_accuracy(self, expected_structure, detected_structure):
        """Calculate accuracy of structure detection"""
        accuracy_scores = {}
        
        # Node count accuracy
        expected_nodes = expected_structure.get('expected_nodes', 0)
        detected_nodes = detected_structure.get('estimated_nodes', 0)
        
        if expected_nodes > 0:
            node_accuracy = 1 - abs(detected_nodes - expected_nodes) / expected_nodes
            node_accuracy = max(0, min(1, node_accuracy))
        else:
            node_accuracy = 0.5  # Default if unknown
        
        accuracy_scores['node_accuracy'] = round(node_accuracy * 100, 2)
        
        # Element type accuracy
        elements = ['has_start', 'has_decisions', 'has_processes', 'has_io']
        element_score = 0
        
        for element in elements:
            if element in expected_structure and element in detected_structure:
                if expected_structure[element] == detected_structure[element]:
                    element_score += 25  # 25% for each correct element
        
        accuracy_scores['element_accuracy'] = element_score
        
        # Overall accuracy (weighted)
        overall = (node_accuracy * 0.6 + (element_score / 100) * 0.4) * 100
        accuracy_scores['overall_accuracy'] = round(overall, 2)
        
        return accuracy_scores
    
    def generate_fr_with_accuracy(self, mermaid_code, extracted_text, description, accuracy_metrics):
        """Generate FR with accuracy analysis - IMPROVED VERSION"""
        
        if not self.ollama_running:
            print("⚠️ Ollama not running. Using simple FR generation.")
            return self.generate_simple_fr(mermaid_code, extracted_text)
        
        # Check if we have any models available
        if not self.available_models:
            print("⚠️ No Ollama models available. Using simple FR generation.")
            return self.generate_simple_fr(mermaid_code, extracted_text)
        
        # Prepare enhanced prompt with accuracy context
        prompt = f"""
        You are a Senior Business Analyst with 15+ years of experience. Based on the following flowchart analysis, generate comprehensive, detailed, and dynamic Functional Requirements.

        IMPORTANT: DO NOT use template-based or pattern-based responses. Each requirement should be unique and context-specific.

        ACCURACY ANALYSIS:
        - Text Extraction Accuracy: {accuracy_metrics.get('text_accuracy', 'N/A')}%
        - Structure Detection Accuracy: {accuracy_metrics.get('structure_accuracy', 'N/A')}%
        - Overall Processing Quality: {accuracy_metrics.get('overall_accuracy', 'N/A')}%

        EXTRACTED TEXT FROM FLOWCHART:
        {extracted_text[:1500]}

        GENERATED MERMAID FLOWCHART REPRESENTATION:
        {mermaid_code}

        STRUCTURE ANALYSIS:
        {description}

        Please provide DETAILED, SPECIFIC, and CONTEXTUAL Functional Requirements. Do NOT use generic patterns like "System shall implement 'X' functionality".

        Instead, create requirements that:
        1. Are specific to the business process shown in the flowchart
        2. Include acceptance criteria where appropriate
        3. Reference specific actors/roles identified in the flowchart
        4. Include data validation rules if applicable
        5. Consider error handling and edge cases
        6. Are measurable and testable

        Format your response as:

        ======================================
        DETAILED FUNCTIONAL REQUIREMENTS
        ======================================

        1. SYSTEM OVERVIEW
        [Brief description of the system based on flowchart]

        2. ACTOR/ROLE DEFINITIONS
        [List and describe each actor/role from the flowchart]

        3. CORE FUNCTIONAL REQUIREMENTS
        [Numbered requirements FR-001 to FR-XXX with detailed descriptions]

        4. BUSINESS RULES
        [Specific rules derived from the flowchart]

        5. DATA REQUIREMENTS
        [Data validation, storage, and processing requirements]

        6. USER INTERFACE REQUIREMENTS (if applicable)
        [UI requirements based on the process flow]

        7. INTEGRATION REQUIREMENTS (if applicable)
        [Integration points with other systems]

        8. SECURITY REQUIREMENTS (if applicable)
        [Access control and security considerations]

        Make each requirement UNIQUE, SPECIFIC, and ACTIONABLE. Avoid generic templates.
        """
        
        try:
            print(f"🤖 Generating FR with {self.model_name}...")
            start_time = time.time()
            
            # Try with smaller context and reduced parameters to save memory
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are an expert Business Analyst who creates detailed, specific, and non-template-based functional requirements from flowcharts.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.3,  # Lower temperature for more focused output
                    'num_predict': 2048,  # Reduced from 4096
                    'top_k': 20,          # Reduced from 40
                    'top_p': 0.8,         # Reduced from 0.9
                    'num_ctx': 2048       # Smaller context window
                }
            )
            
            processing_time = time.time() - start_time
            
            fr_content = response['message']['content']
            
            # Track performance
            perf_metric = {
                'timestamp': datetime.now().isoformat(),
                'model': self.model_name,
                'processing_time': round(processing_time, 2),
                'input_length': len(prompt),
                'output_length': len(fr_content),
                'accuracy_score': accuracy_metrics.get('overall_accuracy', 0)
            }
            self.performance_metrics.append(perf_metric)
            
            print(f"✅ FR generated successfully in {processing_time:.2f}s")
            return fr_content
            
        except Exception as e:
            print(f"⚠️ AI FR generation failed: {e}")
            
            # Try with an even simpler approach if memory is the issue
            if "memory" in str(e).lower() or "4.5" in str(e):
                print("💡 Memory constraint detected. Trying ultra-light generation...")
                return self.generate_ultra_light_fr(mermaid_code, extracted_text)
            
            return self.generate_simple_fr(mermaid_code, extracted_text)
    
    def generate_ultra_light_fr(self, mermaid_code, extracted_text):
        """Ultra-light FR generation for memory-constrained systems"""
        print("🔄 Using ultra-light FR generation...")
        
        # Extract meaningful nodes from mermaid code
        nodes = re.findall(r'(\w+)\[', mermaid_code)
        unique_nodes = list(set([n for n in nodes if n not in ['Start', 'End']]))
        
        # Create more meaningful requirements based on node names
        fr = f"""
        ======================================
        FUNCTIONAL REQUIREMENTS - LIGHT VERSION
        ======================================

        1. SYSTEM OVERVIEW
        Based on the flowchart analysis, this system involves {len(unique_nodes)} key processes 
        related to: {', '.join(unique_nodes[:5])}{'...' if len(unique_nodes) > 5 else ''}

        2. CORE FUNCTIONAL REQUIREMENTS
        """
        
        for i, node in enumerate(unique_nodes[:10], 1):
            # Create more descriptive requirements
            clean_node = node.replace('_', ' ').title()
            fr += f"\nFR-{i:03d}: The system shall provide functionality to handle '{clean_node}' process"
            
            # Add some context-specific details
            if 'order' in node.lower():
                fr += ", including validation of order details and customer information."
            elif 'customer' in node.lower():
                fr += ", with proper customer data management and communication capabilities."
            elif 'status' in node.lower():
                fr += ", ensuring real-time tracking and updates of process status."
            elif 'notify' in node.lower():
                fr += ", with configurable notification channels and templates."
            elif 'review' in node.lower():
                fr += ", including approval workflows and audit trails."
            elif 'accept' in node.lower() or 'reject' in node.lower():
                fr += ", with clear criteria and automated decision support."
            else:
                fr += ", as defined in the business process workflow."
        
        fr += f"""

        3. BUSINESS RULES
        • All processes must maintain audit trails of actions taken
        • Customer communications must be logged and timestamped
        • Order status transitions must follow the defined workflow
        • User roles and permissions must be enforced as per organizational policies

        4. DATA REQUIREMENTS
        • All business data must be validated before processing
        • Customer information must be encrypted at rest and in transit
        • Process status must be updated in real-time
        • Historical data must be retained for regulatory compliance

        5. NOTES
        • This is a lightweight requirements draft based on automated analysis
        • Manual review and refinement required for implementation
        • Consider integrating with existing enterprise systems
        """
        
        return fr
    
    def generate_simple_fr(self, mermaid_code, extracted_text):
        """Fallback FR generation - IMPROVED VERSION"""
        print("🔄 Using improved simple FR generation...")
        
        nodes = re.findall(r'(\w+)\[', mermaid_code)
        unique_nodes = list(set([n for n in nodes if n not in ['Start', 'End']]))
        
        fr = f"""
        ======================================
        FUNCTIONAL REQUIREMENTS (Automated Draft)
        ======================================
        
        1. EXECUTIVE SUMMARY
        • Flowchart analysis identified {len(unique_nodes)} key business processes
        • Automated requirements generation with {len(extracted_text)} characters of extracted text
        • Requires business analyst review and refinement
        
        2. IDENTIFIED PROCESSES
        """
        
        for i, node in enumerate(unique_nodes[:15], 1):
            clean_node = node.replace('_', ' ').title()
            fr += f"{i}. {clean_node}\n"
        
        fr += """
        
        3. RECOMMENDED REQUIREMENTS APPROACH
        
        3.1 Process Flow Requirements
        Consider requirements for:
        • End-to-end process automation
        • Exception handling and error recovery
        • User role-based access control
        • Audit trails and compliance logging
        
        3.2 Data Management Requirements
        • Data validation rules for each process step
        • Database schema for process state management
        • Reporting and analytics capabilities
        • Data retention and archiving policies
        
        3.3 Integration Requirements
        • API specifications for system integration
        • External system interface definitions
        • Batch processing capabilities
        • Real-time data synchronization
        
        4. VALIDATION NOTES
        • Review against original business process documentation
        • Validate with subject matter experts
        • Conduct requirements workshops for refinement
        • Create traceability matrix to business objectives
        """
        
        return fr
    
    def get_performance_report(self):
        """Generate performance analysis report"""
        if not self.performance_metrics:
            return "No performance data available."
        
        df = pd.DataFrame(self.performance_metrics)
        
        report = "🤖 OLLAMA PERFORMANCE REPORT\n"
        report += "=" * 40 + "\n\n"
        
        report += f"Model Used: {self.model_name}\n"
        report += f"Total Requests: {len(df)}\n"
        
        if len(df) > 0:
            report += f"\nPerformance Metrics:\n"
            report += f"• Avg Processing Time: {df['processing_time'].mean():.2f}s\n"
            report += f"• Min Processing Time: {df['processing_time'].min():.2f}s\n"
            report += f"• Max Processing Time: {df['processing_time'].max():.2f}s\n"
            report += f"• Avg Accuracy Score: {df['accuracy_score'].mean():.2f}%\n"
            report += f"• Total Output: {df['output_length'].sum():,} characters\n"
        
        return report

class AdvancedFlowchartAnalyzer:
    """Enhanced flowchart analysis with accuracy tracking"""
    
    def __init__(self):
        self.analysis_history = []
        self.accuracy_benchmarks = []
        
    def analyze_text_quality(self, text):
        """Analyze quality of extracted text"""
        analysis = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'line_count': len([l for l in text.split('\n') if l.strip()]),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text.split() else 0,
            'unique_words': len(set(text.lower().split())),
            'has_flowchart_keywords': self.check_flowchart_keywords(text)
        }
        
        # Calculate quality score (0-100)
        quality_score = 0
        
        # Word count contributes 30%
        if analysis['word_count'] > 10:
            quality_score += 30
        elif analysis['word_count'] > 5:
            quality_score += 15
        
        # Unique words contribute 20%
        if analysis['unique_words'] / max(analysis['word_count'], 1) > 0.5:
            quality_score += 20
        
        # Flowchart keywords contribute 50%
        if analysis['has_flowchart_keywords']:
            quality_score += 50
        
        analysis['quality_score'] = min(100, quality_score)
        
        return analysis
    
    def check_flowchart_keywords(self, text):
        """Check if text contains flowchart-related keywords"""
        keywords = [
            'start', 'end', 'process', 'decision', 'input', 'output',
            'yes', 'no', 'if', 'then', 'else', 'loop', 'repeat',
            'check', 'verify', 'validate', 'calculate', 'compute'
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in keywords if kw in text_lower]
        
        return len(found_keywords) >= 3  # At least 3 keywords
    
    def analyze_mermaid_quality(self, mermaid_code):
        """Analyze quality of generated Mermaid code"""
        analysis = {
            'node_count': len(re.findall(r'(\w+)\[', mermaid_code)),
            'edge_count': len(re.findall(r'-->', mermaid_code)),
            'has_start': 'Start[' in mermaid_code,
            'has_end': 'End[' in mermaid_code,
            'has_decisions': '{' in mermaid_code,
            'syntax_valid': self.validate_mermaid_syntax(mermaid_code)
        }
        
        # Calculate structure score
        structure_score = 0
        if analysis['node_count'] >= 3:
            structure_score += 40
        if analysis['edge_count'] >= 2:
            structure_score += 30
        if analysis['has_start'] and analysis['has_end']:
            structure_score += 20
        if analysis['syntax_valid']:
            structure_score += 10
        
        analysis['structure_score'] = min(100, structure_score)
        
        return analysis
    
    def validate_mermaid_syntax(self, mermaid_code):
        """Validate Mermaid syntax"""
        try:
            # Basic syntax checks
            if not mermaid_code.startswith('graph'):
                return False
            
            lines = mermaid_code.split('\n')
            if len(lines) < 2:
                return False
            
            # Check for node definitions and connections
            has_nodes = any('[' in line or '{' in line for line in lines)
            has_connections = any('-->' in line for line in lines)
            
            return has_nodes and has_connections
        except:
            return False
    
    def generate_accuracy_report(self, text_analysis, mermaid_analysis, ollama_accuracy):
        """Generate comprehensive accuracy report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'text_quality': text_analysis['quality_score'],
            'structure_quality': mermaid_analysis['structure_score'],
            'ollama_accuracy': ollama_accuracy.get('overall_accuracy', 0),
            'details': {
                'text_analysis': text_analysis,
                'mermaid_analysis': mermaid_analysis,
                'ollama_metrics': ollama_accuracy
            }
        }
        
        # Calculate overall accuracy (weighted average)
        weights = {
            'text_quality': 0.3,
            'structure_quality': 0.4,
            'ollama_accuracy': 0.3
        }
        
        overall_score = (
            text_analysis['quality_score'] * weights['text_quality'] +
            mermaid_analysis['structure_score'] * weights['structure_quality'] +
            ollama_accuracy.get('overall_accuracy', 0) * weights['ollama_accuracy']
        )
        
        report['overall_accuracy'] = round(overall_score, 2)
        
        # Add to history
        self.analysis_history.append(report)
        
        return report
    
    def get_accuracy_summary(self):
        """Get summary of all accuracy analyses"""
        if not self.analysis_history:
            return None
        
        summaries = []
        for i, report in enumerate(self.analysis_history, 1):
            summary = {
                'Analysis': f'Run #{i}',
                'Overall Accuracy': f"{report['overall_accuracy']}%",
                'Text Quality': f"{report['text_quality']}%",
                'Structure Quality': f"{report['structure_quality']}%",
                'FR Accuracy': f"{report['ollama_accuracy']}%",
                'Timestamp': report['timestamp'][:19]
            }
            summaries.append(summary)
        
        return summaries

class PDFReportGenerator:
    """Generate comprehensive PDF reports"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for PDF - FIXED VERSION"""
        # Use existing styles instead of trying to add new ones with same names
        # We'll modify existing styles instead
        try:
            # Modify existing Title style
            self.styles['Title'].fontSize = 24
            self.styles['Title'].textColor = colors.HexColor('#2E86AB')
            self.styles['Title'].spaceAfter = 30
        except:
            pass
        
        try:
            # Modify existing Heading1 style
            self.styles['Heading1'].fontSize = 18
            self.styles['Heading1'].textColor = colors.HexColor('#A23B72')
            self.styles['Heading1'].spaceAfter = 12
        except:
            pass
        
        try:
            # Modify existing Heading2 style
            self.styles['Heading2'].fontSize = 14
            self.styles['Heading2'].textColor = colors.HexColor('#F18F01')
            self.styles['Heading2'].spaceAfter = 8
        except:
            pass
        
        # Add only NEW styles that don't exist
        style_names = ['CustomTitle', 'CustomHeading1', 'CustomHeading2', 'BodyText', 'Highlight']
        
        for style_name in style_names:
            if style_name not in self.styles:
                if style_name == 'CustomTitle':
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles['Title'],
                        fontSize=24,
                        textColor=colors.HexColor('#2E86AB'),
                        spaceAfter=30
                    ))
                elif style_name == 'CustomHeading1':
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles['Heading1'],
                        fontSize=18,
                        textColor=colors.HexColor('#A23B72'),
                        spaceAfter=12
                    ))
                elif style_name == 'CustomHeading2':
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles['Heading2'],
                        fontSize=14,
                        textColor=colors.HexColor('#F18F01'),
                        spaceAfter=8
                    ))
                elif style_name == 'BodyText':
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles['Normal'],
                        fontSize=10,
                        spaceAfter=6
                    ))
                elif style_name == 'Highlight':
                    self.styles.add(ParagraphStyle(
                        name=style_name,
                        parent=self.styles['Normal'],
                        fontSize=10,
                        backColor=colors.HexColor('#F0F8FF'),
                        borderPadding=5,
                        borderColor=colors.HexColor('#2E86AB'),
                        borderWidth=1
                    ))
    
    def create_report(self, results, accuracy_report, output_path):
        """Create comprehensive PDF report"""
        try:
            doc = SimpleDocTemplate(
                output_path,
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Title Page
            story.append(Paragraph("Usecase to FR REPORT", self.styles['CustomTitle'] if 'CustomTitle' in self.styles else self.styles['Title']))
            story.append(Spacer(1, 20))
            
            # Report metadata
            metadata = f"""
            <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Total Files Processed:</b> {len(results)}<br/>
            <b>Overall Accuracy:</b> {accuracy_report.get('overall_accuracy', 'N/A')}%<br/>
            <b>Report ID:</b> {datetime.now().strftime('%Y%m%d_%H%M%S')}
            """
            story.append(Paragraph(metadata, self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
            story.append(PageBreak())
            
            # Table of Contents
            story.append(Paragraph("TABLE OF CONTENTS", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            toc = [
                ["1. Executive Summary", "2"],
                ["2. Accuracy Analysis", "3"],
                ["3. File-by-File Results", "4"],
                ["4. Functional Requirements", "5"],
                ["5. Recommendations", "6"]
            ]
            
            toc_table = Table(toc, colWidths=[4*inch, 1*inch])
            toc_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(toc_table)
            story.append(PageBreak())
            
            # 1. Executive Summary
            story.append(Paragraph("1. EXECUTIVE SUMMARY", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            
            exec_summary = f"""
            This report summarizes the conversion of {len(results)} flowchart files into Functional Requirements.
            The automated pipeline achieved an overall accuracy of <b>{accuracy_report.get('overall_accuracy', 'N/A')}%</b>.
            
            <b>Key Findings:</b><br/>
            • Text extraction quality: {accuracy_report.get('text_quality', 'N/A')}%<br/>
            • Structure detection quality: {accuracy_report.get('structure_quality', 'N/A')}%<br/>
            • FR generation accuracy: {accuracy_report.get('ollama_accuracy', 'N/A')}%<br/>
            
            <b>Processing Details:</b><br/>
            • Files processed: {len(results)}<br/>
            • Model used: Mistral<br/>
            • Generation method: AI-assisted with accuracy validation<br/>
            """
            story.append(Paragraph(exec_summary, self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 2. Accuracy Analysis
            story.append(Paragraph("2. ACCURACY ANALYSIS", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            
            accuracy_details = f"""
            <b>Overall Accuracy Score:</b> {accuracy_report.get('overall_accuracy', 'N/A')}%<br/>
            <b>Breakdown by Component:</b><br/>
            • Text Extraction: {accuracy_report.get('text_quality', 'N/A')}%<br/>
            • Structure Detection: {accuracy_report.get('structure_quality', 'N/A')}%<br/>
            • FR Generation: {accuracy_report.get('ollama_accuracy', 'N/A')}%<br/>
            
            <b>Quality Assessment:</b><br/>
            """
            
            overall_acc = accuracy_report.get('overall_accuracy', 0)
            if overall_acc >= 80:
                accuracy_details += "✓ <b>EXCELLENT</b> - High confidence in generated requirements<br/>"
            elif overall_acc >= 60:
                accuracy_details += "✓ <b>GOOD</b> - Requirements are reliable with minor review needed<br/>"
            elif overall_acc >= 40:
                accuracy_details += "⚠ <b>FAIR</b> - Manual review and validation recommended<br/>"
            else:
                accuracy_details += "⚠ <b>LIMITED</b> - Significant manual intervention required<br/>"
            
            story.append(Paragraph(accuracy_details, self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
            story.append(Spacer(1, 20))
            
            # 3. File-by-File Results
            story.append(Paragraph("3. FILE-BY-FILE RESULTS", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            
            for i, result in enumerate(results, 1):
                story.append(Paragraph(f"3.{i} {result['file_name']}", self.styles['CustomHeading2'] if 'CustomHeading2' in self.styles else self.styles['Heading2']))
                
                file_summary = f"""
                <b>File Type:</b> {result['file_type'].upper()}<br/>
                <b>Processing Time:</b> {result.get('processing_time', 'N/A')}<br/>
                <b>Accuracy Score:</b> {result.get('accuracy_score', 'N/A')}%<br/>
                <b>Text Extraction:</b> {len(result.get('extracted_text', ''))} characters extracted<br/>
                <b>Flowchart Nodes:</b> {result.get('node_count', 'N/A')}<br/>
                """
                story.append(Paragraph(file_summary, self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
                
                # Add Mermaid code in a box
                story.append(Paragraph("<b>Generated Mermaid Code:</b>", self.styles['CustomHeading2'] if 'CustomHeading2' in self.styles else self.styles['Heading2']))
                mermaid_box = f"<font face='Courier' size='8'>{result.get('mermaid_code', 'No code generated')}</font>"
                story.append(Paragraph(mermaid_box, self.styles['Highlight'] if 'Highlight' in self.styles else self.styles['Normal']))
                
                story.append(Spacer(1, 12))
            
            story.append(PageBreak())
            
            # 4. Functional Requirements
            story.append(Paragraph("4. FUNCTIONAL REQUIREMENTS", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            
            for i, result in enumerate(results, 1):
                if result.get('fr_content'):
                    story.append(Paragraph(f"4.{i} Requirements for: {result['file_name']}", self.styles['CustomHeading2'] if 'CustomHeading2' in self.styles else self.styles['Heading2']))
                    
                    # Clean FR content for PDF - handle different formats
                    fr_content = result['fr_content']
                    
                    # Replace common markdown/bullet characters for better PDF rendering
                    fr_content = fr_content.replace('•', '✓').replace('*', '•').replace('#', '').replace('=', '')
                    
                    # Split into paragraphs and add each
                    fr_paragraphs = fr_content.split('\n')
                    
                    for para in fr_paragraphs[:100]:  # Limit to first 100 paragraphs
                        if para.strip():
                            # Check if this looks like a heading
                            if para.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')) and len(para.strip()) < 100:
                                story.append(Paragraph(f"<b>{para.strip()}</b>", self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
                            elif any(keyword in para.lower() for keyword in ['requirement', 'fr-', 'shall', 'must']):
                                story.append(Paragraph(f"<b>{para.strip()}</b>", self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
                            else:
                                story.append(Paragraph(para.strip(), self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
                    
                    story.append(Spacer(1, 20))
            
            # 5. Recommendations
            story.append(Paragraph("5. RECOMMENDATIONS", self.styles['CustomHeading1'] if 'CustomHeading1' in self.styles else self.styles['Heading1']))
            
            recommendations = """
            <b>Based on the analysis, the following recommendations are provided:</b><br/><br/>
            
            <b>1. VALIDATION REQUIREMENTS:</b><br/>
            ✓ Review all generated requirements against original flowcharts<br/>
            ✓ Validate decision points and branching logic<br/>
            ✓ Verify data flow and input/output specifications<br/><br/>
            
            <b>2. IMPLEMENTATION PRIORITIES:</b><br/>
            ✓ Focus on requirements with highest accuracy scores first<br/>
            ✓ Schedule manual review for requirements below 60% accuracy<br/>
            ✓ Consider re-processing low-accuracy files with manual corrections<br/><br/>
            
            <b>3. QUALITY IMPROVEMENTS:</b><br/>
            ✓ Provide clearer source images with higher resolution<br/>
            ✓ Use standardized flowchart symbols and notations<br/>
            ✓ Include descriptive text within flowchart elements<br/><br/>
            
            <b>4. NEXT STEPS:</b><br/>
            ✓ Schedule stakeholder review session<br/>
            ✓ Create traceability matrix<br/>
            ✓ Develop test cases based on generated requirements<br/>
            """
            story.append(Paragraph(recommendations, self.styles['BodyText'] if 'BodyText' in self.styles else self.styles['Normal']))
            
            # Build PDF
            doc.build(story)
            return output_path
            
        except Exception as e:
            print(f"Error creating PDF: {e}")
            raise

class WindowsFlowchartConverter:
    """Windows GUI version of the flowchart converter"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="flowchart_fr_")
        self.setup_directories()
        
        # Initialize components
        self.ollama_manager = AdvancedOllamaManager(model_name="mistral")
        self.accuracy_analyzer = AdvancedFlowchartAnalyzer()
        self.pdf_generator = PDFReportGenerator()
        
        # Results storage
        self.all_results = []
        self.current_results = []
        
        # Create GUI
        self.create_gui()
        
    def setup_directories(self):
        """Create directory structure"""
        self.input_dir = os.path.join(self.temp_dir, "input")
        self.output_dir = os.path.join(self.temp_dir, "output")
        self.reports_dir = os.path.join(self.temp_dir, "reports")
        
        for dir_path in [self.input_dir, self.output_dir, self.reports_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def create_gui(self):
        """Create Windows GUI"""
        self.root = tk.Tk()
        self.root.title("Flowchart to Functional Requirements Converter")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = tk.Label(main_frame, 
                              text="🚀 Usecase to Functional Requirement", 
                              font=("Arial", 20, "bold"),
                              bg='#f0f0f0',
                              fg='#2E86AB')
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Button Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=1, column=0, pady=(0, 20))
        
        # Buttons
        self.upload_btn = ttk.Button(button_frame, 
                                    text="📤 Upload & Process Files", 
                                    command=self.upload_files,
                                    width=25)
        self.upload_btn.grid(row=0, column=0, padx=5)
        
        self.download_btn = ttk.Button(button_frame, 
                                      text="📥 Generate PDF Report", 
                                      command=self.generate_pdf,
                                      width=25)
        self.download_btn.grid(row=0, column=1, padx=5)
        
        self.view_btn = ttk.Button(button_frame, 
                                  text="📊 View Results", 
                                  command=self.view_results,
                                  width=25)
        self.view_btn.grid(row=0, column=2, padx=5)
        
        self.clear_btn = ttk.Button(button_frame, 
                                   text="🔄 Clear Results", 
                                   command=self.clear_results,
                                   width=25)
        self.clear_btn.grid(row=0, column=3, padx=5)
        
        # Model selection
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=2, column=0, pady=(0, 10))
        
        tk.Label(model_frame, text="Ollama Model:", font=("Arial", 9), bg='#f0f0f0').grid(row=0, column=0, padx=(0, 5))
        
        self.model_var = tk.StringVar(value="mistral")
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=20, state="readonly")
        self.model_combo.grid(row=0, column=1, padx=(0, 10))
        self.model_combo['values'] = ['mistral', 'llama2', 'codellama', 'neural-chat']
        
        self.refresh_models_btn = ttk.Button(model_frame, 
                                           text="🔄 Refresh Models", 
                                           command=self.refresh_models,
                                           width=15)
        self.refresh_models_btn.grid(row=0, column=2)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, 
                                           variable=self.progress_var,
                                           maximum=100,
                                           length=800)
        self.progress_bar.grid(row=3, column=0, pady=(0, 10))
        
        # Status label
        self.status_label = tk.Label(main_frame, 
                                    text="Ready to process files...", 
                                    font=("Arial", 10),
                                    bg='#f0f0f0')
        self.status_label.grid(row=4, column=0, pady=(0, 20))
        
        # Results text area with scrollbar
        results_frame = ttk.Frame(main_frame)
        results_frame.grid(row=5, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Text widget for results
        text_scrollbar = ttk.Scrollbar(results_frame)
        text_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        self.results_text = tk.Text(results_frame, 
                                   wrap=tk.WORD, 
                                   yscrollcommand=text_scrollbar.set,
                                   height=20,
                                   width=100)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scrollbar.config(command=self.results_text.yview)
        
        # Configure tags for text coloring
        self.results_text.tag_configure("success", foreground="green")
        self.results_text.tag_configure("error", foreground="red")
        self.results_text.tag_configure("warning", foreground="orange")
        self.results_text.tag_configure("info", foreground="blue")
        self.results_text.tag_configure("highlight", background="lightyellow")
        
        # File counter
        self.file_counter = tk.Label(main_frame, 
                                    text="Files processed: 0", 
                                    font=("Arial", 9),
                                    bg='#f0f0f0')
        self.file_counter.grid(row=6, column=0, pady=(10, 0))
        
        # Configure main frame grid weights
        main_frame.rowconfigure(5, weight=1)
        
    def refresh_models(self):
        """Refresh available Ollama models"""
        try:
            self.update_status("🔄 Refreshing available models...", "info")
            self.ollama_manager.available_models = self.ollama_manager.get_available_models()
            
            if self.ollama_manager.available_models:
                # Update combo box with available models
                models = [model.split(':')[0] for model in self.ollama_manager.available_models]
                self.model_combo['values'] = list(set(models))  # Remove duplicates
                self.update_status(f"✅ Available models: {', '.join(models)}", "success")
            else:
                self.update_status("⚠️ No models available. Please pull a model first.", "warning")
                
        except Exception as e:
            self.update_status(f"❌ Error refreshing models: {str(e)}", "error")
    
    def update_status(self, message, status_type="info"):
        """Update status label"""
        self.status_label.config(text=message)
        
        # Add to results text with color coding
        self.results_text.insert(tk.END, f"{message}\n", status_type)
        self.results_text.see(tk.END)
        self.root.update()
    
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update()
    
    def upload_files(self):
        """Handle file upload"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif"),
            ("SVG files", "*.svg"),
            ("ZIP files", "*.zip"),
            ("All files", "*.*")
        ]
        
        filenames = filedialog.askopenfilenames(
            title="Select flowchart files",
            filetypes=filetypes
        )
        
        if not filenames:
            return
        
        # Update model in ollama manager
        selected_model = self.model_var.get()
        if selected_model != self.ollama_manager.model_name:
            self.ollama_manager.model_name = selected_model
            self.update_status(f"📦 Using model: {selected_model}", "info")
        
        # Process in background thread
        threading.Thread(target=self.process_files, args=(filenames,), daemon=True).start()
    
    def process_files(self, filenames):
        """Process uploaded files"""
        try:
            self.update_status(f"📋 Processing {len(filenames)} file(s)...")
            self.update_progress(0)
            
            self.current_results = []
            total_files = len(filenames)
            
            for idx, filename in enumerate(filenames, 1):
                self.update_status(f"🔍 Processing: {os.path.basename(filename)}")
                self.update_progress((idx-1)/total_files * 100)
                
                # Process single file
                result = self.process_single_file(filename)
                if result:
                    self.current_results.append(result)
                    self.update_status(f"   ✅ Extracted {len(result.get('full_text', ''))} characters", "success")
                    self.update_status(f"   📊 Accuracy: {result['accuracy_score']}%", "success")
                
                self.update_progress(idx/total_files * 100)
            
            # Add to all results
            if self.current_results:
                self.all_results.extend(self.current_results)
                self.update_status(f"✅ Successfully processed {len(self.current_results)} file(s)", "success")
                self.update_status(f"📊 Total files processed: {len(self.all_results)}")
                self.file_counter.config(text=f"Files processed: {len(self.all_results)}")
                
                # Display results
                self.display_results()
            
            self.update_progress(100)
            self.update_status("✅ Processing complete!", "success")
            
        except Exception as e:
            self.update_status(f"❌ Error: {str(e)}", "error")
            import traceback
            traceback.print_exc()
    
    def extract_svg_text(self, svg_path):
        """Extract text from SVG"""
        try:
            with open(svg_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract all text elements
            text_patterns = [r'<text[^>]*>(.*?)</text>', r'<tspan[^>]*>(.*?)</tspan>']
            all_text = []
            
            for pattern in text_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    if isinstance(match, str):
                        clean_text = re.sub(r'<[^>]+>', '', match).strip()
                        if clean_text:
                            all_text.append(clean_text)
            
            return '\n'.join(all_text[:100])  # Limit to 100 text elements
            
        except Exception as e:
            print(f"SVG extraction error: {e}")
            return ""
    
    def extract_image_text(self, image_path):
        """Extract text from image using OCR"""
        try:
            # Read and preprocess image
            img = cv2.imread(image_path)
            if img is None:
                return ""
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply preprocessing
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
            
            # Use pytesseract
            text = pytesseract.image_to_string(processed, config='--oem 3 --psm 6')
            return text.strip()
            
        except Exception as e:
            print(f"OCR error: {e}")
            return ""
    
    def analyze_structure(self, text):
        """Analyze flowchart structure"""
        structure = {
            'has_start': any(word in text.lower() for word in ['start', 'begin']),
            'has_end': any(word in text.lower() for word in ['end', 'stop', 'finish']),
            'has_decisions': any(word in text.lower() for word in ['if', 'decision', 'check']),
            'has_processes': any(word in text.lower() for word in ['process', 'calculate', 'compute']),
            'estimated_nodes': len([line for line in text.split('\n') if line.strip()])
        }
        
        return structure
    
    def generate_mermaid(self, text, structure):
        """Generate Mermaid code"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return "graph TD\n    Start[Start] --> End[End]"
        
        # Clean node names
        nodes = []
        for line in lines[:15]:  # Limit nodes
            clean_name = re.sub(r'[^\w\s]', '', line)
            clean_name = clean_name.replace(' ', '_')[:20]
            if clean_name and clean_name not in ['Start', 'End']:
                nodes.append(clean_name)
        
        # Generate Mermaid
        mermaid = "graph TD\n"
        mermaid += "    Start[Start]\n"
        
        for node in nodes:
            mermaid += f"    {node}[{node}]\n"
        
        # Connect nodes
        mermaid += f"    Start --> {nodes[0] if nodes else 'Process'}\n"
        for i in range(len(nodes) - 1):
            mermaid += f"    {nodes[i]} --> {nodes[i+1]}\n"
        
        if nodes:
            mermaid += f"    {nodes[-1]} --> End[End]\n"
        
        return mermaid
    
    def generate_description(self, text, structure):
        """Generate description"""
        desc = f"Flowchart Analysis:\n"
        desc += f"• Text length: {len(text)} characters\n"
        desc += f"• Estimated nodes: {structure['estimated_nodes']}\n"
        desc += f"• Has decisions: {structure['has_decisions']}\n"
        desc += f"• Has start/end: {structure['has_start']}/{structure['has_end']}\n"
        return desc
    
    def process_single_file(self, file_path):
        """Process a single file"""
        start_time = time.time()
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        print(f"  🔍 Processing: {file_name}")
        
        try:
            # Extract text
            if file_ext == '.svg':
                extracted_text = self.extract_svg_text(file_path)
            else:
                extracted_text = self.extract_image_text(file_path)
            
            print(f"    Extracted {len(extracted_text)} characters")
            
            # Analyze text quality
            text_analysis = self.accuracy_analyzer.analyze_text_quality(extracted_text)
            
            # Generate Mermaid
            structure = self.analyze_structure(extracted_text)
            mermaid_code = self.generate_mermaid(extracted_text, structure)
            
            # Analyze Mermaid quality
            mermaid_analysis = self.accuracy_analyzer.analyze_mermaid_quality(mermaid_code)
            
            # Calculate accuracy metrics
            accuracy_metrics = {
                'text_accuracy': text_analysis['quality_score'],
                'structure_accuracy': mermaid_analysis['structure_score'],
                'overall_accuracy': (text_analysis['quality_score'] * 0.3 + 
                                    mermaid_analysis['structure_score'] * 0.7)
            }
            
            # Generate FR with accuracy context
            description = self.generate_description(extracted_text, structure)
            fr_content = self.ollama_manager.generate_fr_with_accuracy(
                mermaid_code, extracted_text, description, accuracy_metrics
            )
            
            # Generate accuracy report
            accuracy_report = self.accuracy_analyzer.generate_accuracy_report(
                text_analysis, mermaid_analysis, accuracy_metrics
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'file_name': file_name,
                'file_type': file_ext,
                'processing_time': f"{processing_time:.2f}s",
                'extracted_text': extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""),
                'full_text': extracted_text,
                'mermaid_code': mermaid_code,
                'fr_content': fr_content,
                'accuracy_score': accuracy_report['overall_accuracy'],
                'accuracy_details': accuracy_report,
                'text_analysis': text_analysis,
                'mermaid_analysis': mermaid_analysis,
                'node_count': mermaid_analysis['node_count']
            }
            
            print(f"    ✅ Accuracy: {accuracy_report['overall_accuracy']}%")
            print(f"    📝 Generated FR: {len(fr_content)} characters")
            
            # Check if we got the simple template-based FR
            if "System shall implement" in fr_content and "functionality" in fr_content:
                print("    ⚠️ Warning: Generated FR appears to be template-based")
            
            return result
            
        except Exception as e:
            print(f"    ❌ Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def display_results(self):
        """Display results in text widget"""
        self.results_text.delete(1.0, tk.END)
        
        if not self.current_results:
            self.results_text.insert(tk.END, "No results to display.")
            return
        
        for result in self.current_results:
            self.results_text.insert(tk.END, f"\n{'='*60}\n", "info")
            self.results_text.insert(tk.END, f"📄 File: {result['file_name']}\n", "info")
            self.results_text.insert(tk.END, f"📊 Accuracy: {result['accuracy_score']}%\n", "success")
            self.results_text.insert(tk.END, f"⏱️ Time: {result['processing_time']}\n", "info")
            self.results_text.insert(tk.END, f"📝 Text extracted: {len(result.get('full_text', ''))} chars\n", "info")
            
            if result.get('fr_content'):
                # Show more of the FR content
                fr_preview = result['fr_content'][:500] + ("..." if len(result['fr_content']) > 500 else "")
                self.results_text.insert(tk.END, f"\n📋 FR Preview:\n", "info")
                self.results_text.insert(tk.END, f"{fr_preview}\n\n", "highlight")
    
    def view_results(self):
        """View all results"""
        if not self.all_results:
            messagebox.showinfo("No Results", "No results to display. Please process files first.")
            return
        
        # Create new window for results
        results_window = tk.Toplevel(self.root)
        results_window.title("All Results")
        results_window.geometry("800x600")
        
        # Create text widget
        text_widget = tk.Text(results_window, wrap=tk.WORD)
        text_widget.pack(expand=True, fill='both')
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_window, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        
        # Display all results
        text_widget.insert(tk.END, f"📈 ALL PROCESSED FILES ({len(self.all_results)} total)\n\n")
        
        for i, result in enumerate(self.all_results, 1):
            text_widget.insert(tk.END, f"{i}. {result['file_name']} - {result['accuracy_score']}% - {result['processing_time']}\n")
    
    def generate_pdf(self):
        """Generate PDF report"""
        if not self.all_results:
            messagebox.showwarning("No Data", "No results to generate PDF. Please process files first.")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"Flowchart_FR_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        
        if not filename:
            return
        
        # Generate PDF in background thread
        threading.Thread(target=self._generate_pdf_thread, args=(filename,), daemon=True).start()
    
    def _generate_pdf_thread(self, filename):
        """Thread for PDF generation"""
        try:
            self.update_status("📝 Generating PDF report...")
            
            # Calculate overall accuracy
            overall_acc = np.mean([r['accuracy_score'] for r in self.all_results]) if self.all_results else 0
            
            # Prepare accuracy report
            accuracy_report = {
                'overall_accuracy': round(overall_acc, 2),
                'text_quality': round(np.mean([r['text_analysis']['quality_score'] for r in self.all_results]), 2),
                'structure_quality': round(np.mean([r['mermaid_analysis']['structure_score'] for r in self.all_results]), 2),
                'ollama_accuracy': round(np.mean([r['accuracy_details']['details']['ollama_metrics'].get('overall_accuracy', 0) 
                                                for r in self.all_results]), 2),
            }
            
            # Generate PDF
            self.pdf_generator.create_report(self.all_results, accuracy_report, filename)
            
            self.update_status(f"✅ PDF saved to: {filename}", "success")
            messagebox.showinfo("Success", f"PDF report saved successfully!\n\n{filename}")
            
        except Exception as e:
            self.update_status(f"❌ PDF generation failed: {str(e)}", "error")
            messagebox.showerror("Error", f"Failed to generate PDF:\n{str(e)}")
    
    def clear_results(self):
        """Clear all results"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear all results?"):
            self.all_results = []
            self.current_results = []
            self.results_text.delete(1.0, tk.END)
            self.file_counter.config(text="Files processed: 0")
            self.update_status("Results cleared. Ready for new files.", "info")
    
    def run(self):
        """Run the application"""
        self.root.mainloop()

def main():
    """Main function"""
    print("=" * 70)
    print("🚀 WINDOWS FLOWCHART TO FUNCTIONAL REQUIREMENTS CONVERTER")
    print("=" * 70)
    print("\n✨ System Requirements:")
    print("• Python 3.9+ installed")
    print("• Tesseract OCR installed (for text extraction)")
    print("• Node.js installed (for Mermaid)")
    print("• Ollama installed with Mistral model")
    print("\n🎯 How to use:")
    print("1. Click 'Upload & Process Files' to select files")
    print("2. View results in the window")
    print("3. Generate PDF report when done")
    print("4. Process multiple batches as needed")
    print("\n💡 Memory Tip: If Mistral fails, try smaller models like 'llama2' or 'neural-chat'")
    print("\n" + "=" * 70)
    
    # Check for Tesseract
    try:
        pytesseract.get_tesseract_version()
        print("✅ Tesseract found")
    except:
        print("⚠️ WARNING: Tesseract OCR not found!")
        print("Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("And update the path in the script if needed.")
    
    # Create and run converter
    converter = WindowsFlowchartConverter()
    converter.run()

if __name__ == "__main__":
    main()