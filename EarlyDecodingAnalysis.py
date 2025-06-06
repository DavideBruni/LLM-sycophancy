import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from scipy import stats
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class EarlyDecodingAnalysis:
    def __init__(self, base_dir="output_earlydecoding"):
        self.base_dir = Path(base_dir)
        self.results = {}
        self.analysis_results = {}
        
    def load_results(self):
        """Load all available result files."""
        print("Loading result files...")
        
        # Find all pkl files
        pkl_files = list(self.base_dir.glob("**/*.pkl"))
        
        for pkl_file in pkl_files:
            try:
                # Parse filename to extract metadata
                filename = pkl_file.stem
                parts = filename.split('_')
                
                # Extract model info and question type
                model_parts = []
                for i, part in enumerate(parts):
                    if part in ['logit', 'cot']:
                        inference_mode = part
                        layer_info = parts[i+1] if i+1 < len(parts) else 'all'
                        break
                    model_parts.append(part)
                
                model_name = '_'.join(model_parts)
                question_type = pkl_file.parent.name
                
                # Load data
                df = pd.read_pickle(pkl_file)
                
                key = f"{model_name}_{question_type}"
                self.results[key] = {
                    'data': df,
                    'model_name': model_name,
                    'question_type': question_type,
                    'inference_mode': inference_mode,
                    'layer_info': layer_info,
                    'file_path': str(pkl_file)
                }
                
                print(f"Loaded: {key} ({len(df)} questions)")
                
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
        
        print(f"Total loaded: {len(self.results)} result sets")
        return self.results
    
    def extract_layer_predictions(self, layer_logits_dict):
        """Extract predictions for each layer from logits."""
        layer_predictions = {}
        
        for layer_name, logits in layer_logits_dict.items():
            if isinstance(logits, dict) and all(letter in logits for letter in 'ABCD'):
                # Convert logits to probabilities and get prediction
                logit_values = [logits[letter] for letter in 'ABCD']
                probs = np.exp(logit_values) / np.sum(np.exp(logit_values))
                prediction = 'ABCD'[np.argmax(probs)]
                confidence = np.max(probs)
                
                layer_predictions[layer_name] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'logits': logits,
                    'probabilities': dict(zip('ABCD', probs))
                }
        
        return layer_predictions
    
    def analyze_early_decoding_patterns(self):
        """Analyze when models start making consistent predictions across layers."""
        print("Analyzing early decoding patterns...")
        
        for key, result_data in self.results.items():
            df = result_data['data']
            model_name = result_data['model_name']
            question_type = result_data['question_type']
            
            print(f"\nAnalyzing {key}...")
            
            # Extract layer-wise predictions for each question
            layer_analysis = []
            
            for idx, row in df.iterrows():
                if pd.isna(row['layer_logits']) or not isinstance(row['layer_logits'], dict):
                    continue
                
                layer_preds = self.extract_layer_predictions(row['layer_logits'])
                
                # Get layer numbers and sort them
                layer_nums = sorted([int(layer.split('_')[1]) for layer in layer_preds.keys()])
                
                # Track prediction stability
                predictions_by_layer = []
                confidences_by_layer = []
                
                for layer_num in layer_nums:
                    layer_key = f"Layer_{layer_num}"
                    if layer_key in layer_preds:
                        predictions_by_layer.append(layer_preds[layer_key]['prediction'])
                        confidences_by_layer.append(layer_preds[layer_key]['confidence'])
                
                # Find when prediction stabilizes
                final_prediction = predictions_by_layer[-1] if predictions_by_layer else None
                stabilization_layer = None
                
                # Find first layer where prediction matches final prediction and stays consistent
                for i in range(len(predictions_by_layer)):
                    if predictions_by_layer[i] == final_prediction:
                        # Check if it stays consistent for at least 3 consecutive layers
                        consistent_count = 1
                        for j in range(i+1, min(i+5, len(predictions_by_layer))):
                            if predictions_by_layer[j] == final_prediction:
                                consistent_count += 1
                            else:
                                break
                        
                        if consistent_count >= 3:
                            stabilization_layer = layer_nums[i]
                            break
                
                layer_analysis.append({
                    'question_idx': idx,
                    'final_prediction': final_prediction,
                    'correct_answer': row.get('answer', None),
                    'stabilization_layer': stabilization_layer,
                    'predictions_by_layer': predictions_by_layer,
                    'confidences_by_layer': confidences_by_layer,
                    'layer_numbers': layer_nums,
                    'is_correct': final_prediction == row.get('answer', None) if row.get('answer') else None
                })
            
            self.analysis_results[key] = {
                'layer_analysis': layer_analysis,
                'model_info': result_data
            }
            
            # Calculate summary statistics
            stabilization_layers = [x['stabilization_layer'] for x in layer_analysis if x['stabilization_layer'] is not None]
            if stabilization_layers:
                print(f"  Average stabilization layer: {np.mean(stabilization_layers):.2f}")
                print(f"  Median stabilization layer: {np.median(stabilization_layers):.2f}")
                print(f"  Questions that stabilize: {len(stabilization_layers)}/{len(layer_analysis)} ({len(stabilization_layers)/len(layer_analysis)*100:.1f}%)")
    
    def compare_plain_vs_opinion(self):
        """Compare early decoding behavior between plain and opinion-only questions."""
        print("\nComparing plain vs opinion-only questions...")
        
        # Find matching model pairs
        plain_keys = [k for k in self.analysis_results.keys() if 'plain' in k]
        opinion_keys = [k for k in self.analysis_results.keys() if 'opinion_only' in k]
        
        comparisons = []
        
        for plain_key in plain_keys:
            model_name = plain_key.replace('_plain', '')
            opinion_key = f"{model_name}_opinion_only"
            
            if opinion_key in opinion_keys:
                comparisons.append((plain_key, opinion_key))
        
        if not comparisons:
            print("No matching plain/opinion pairs found")
            return
        
        comparison_results = {}
        
        for plain_key, opinion_key in comparisons:
            print(f"\nComparing {plain_key} vs {opinion_key}")
            
            plain_analysis = self.analysis_results[plain_key]['layer_analysis']
            opinion_analysis = self.analysis_results[opinion_key]['layer_analysis']
            
            # Extract stabilization layers
            plain_stab = [x['stabilization_layer'] for x in plain_analysis if x['stabilization_layer'] is not None]
            opinion_stab = [x['stabilization_layer'] for x in opinion_analysis if x['stabilization_layer'] is not None]
            
            # Statistical comparison
            if plain_stab and opinion_stab:
                stat, p_value = stats.mannwhitneyu(plain_stab, opinion_stab, alternative='two-sided')
                
                comparison_results[plain_key.split('_')[0]] = {
                    'plain_mean_stab': np.mean(plain_stab),
                    'plain_median_stab': np.median(plain_stab),
                    'opinion_mean_stab': np.mean(opinion_stab),
                    'opinion_median_stab': np.median(opinion_stab),
                    'p_value': p_value,
                    'effect_size': (np.mean(opinion_stab) - np.mean(plain_stab)) / np.std(plain_stab + opinion_stab),
                    'plain_data': plain_stab,
                    'opinion_data': opinion_stab
                }
                
                print(f"  Plain - Mean: {np.mean(plain_stab):.2f}, Median: {np.median(plain_stab):.2f}")
                print(f"  Opinion - Mean: {np.mean(opinion_stab):.2f}, Median: {np.median(opinion_stab):.2f}")
                print(f"  Statistical test p-value: {p_value:.4f}")
                print(f"  Effect size: {comparison_results[plain_key.split('_')[0]]['effect_size']:.3f}")
        
        return comparison_results
    
    def analyze_sycophancy_emergence(self):
        """Analyze when sycophantic behavior emerges in opinion-only questions."""
        print("\nAnalyzing sycophancy emergence...")
        
        opinion_keys = [k for k in self.analysis_results.keys() if 'opinion_only' in k]
        
        for key in opinion_keys:
            print(f"\nAnalyzing sycophancy in {key}")
            
            analysis_data = self.analysis_results[key]['layer_analysis']
            df = self.analysis_results[key]['model_info']['data']
            
            # Analyze layer-wise alignment with opinion (if available)
            sycophancy_analysis = []
            
            for item in analysis_data:
                idx = item['question_idx']
                row = df.iloc[idx] if idx < len(df) else None
                
                if row is None:
                    continue
                
                # If there's an opinion column, analyze alignment
                opinion_answer = None
                if 'opinion' in row:
                    opinion_answer = row['opinion']
                elif 'human_opinion' in row:
                    opinion_answer = row['human_opinion']
                
                if opinion_answer and opinion_answer in 'ABCD':
                    # Track when prediction aligns with opinion across layers
                    layer_nums = item['layer_numbers']
                    predictions = item['predictions_by_layer']
                    
                    opinion_alignment_by_layer = []
                    for pred in predictions:
                        opinion_alignment_by_layer.append(pred == opinion_answer)
                    
                    # Find when sycophancy starts (first consistent alignment with opinion)
                    sycophancy_start_layer = None
                    for i in range(len(opinion_alignment_by_layer)):
                        if opinion_alignment_by_layer[i]:
                            # Check consistency
                            consistent_count = 1
                            for j in range(i+1, min(i+3, len(opinion_alignment_by_layer))):
                                if opinion_alignment_by_layer[j]:
                                    consistent_count += 1
                                else:
                                    break
                            
                            if consistent_count >= 2:
                                sycophancy_start_layer = layer_nums[i]
                                break
                    
                    sycophancy_analysis.append({
                        'question_idx': idx,
                        'opinion_answer': opinion_answer,
                        'correct_answer': row.get('answer', None),
                        'final_prediction': item['final_prediction'],
                        'sycophancy_start_layer': sycophancy_start_layer,
                        'is_sycophantic': item['final_prediction'] == opinion_answer,
                        'is_correct': item['final_prediction'] == row.get('answer', None) if row.get('answer') else None,
                        'layer_numbers': layer_nums,
                        'opinion_alignment_by_layer': opinion_alignment_by_layer
                    })
            
            if sycophancy_analysis:
                sycophantic_questions = [x for x in sycophancy_analysis if x['is_sycophantic']]
                sycophancy_start_layers = [x['sycophancy_start_layer'] for x in sycophantic_questions if x['sycophancy_start_layer'] is not None]
                
                print(f"  Sycophantic responses: {len(sycophantic_questions)}/{len(sycophancy_analysis)} ({len(sycophantic_questions)/len(sycophancy_analysis)*100:.1f}%)")
                if sycophancy_start_layers:
                    print(f"  Average sycophancy emergence layer: {np.mean(sycophancy_start_layers):.2f}")
                    print(f"  Median sycophancy emergence layer: {np.median(sycophancy_start_layers):.2f}")
                
                self.analysis_results[key]['sycophancy_analysis'] = sycophancy_analysis
    
    def create_visualizations(self):
        """Create comprehensive visualizations of the analysis."""
        print("\nCreating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Stabilization layer distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Early Decoding Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Stabilization layer distribution by model and question type
        ax1 = axes[0, 0]
        stabilization_data = []
        
        for key, analysis in self.analysis_results.items():
            layer_analysis = analysis['layer_analysis']
            stab_layers = [x['stabilization_layer'] for x in layer_analysis if x['stabilization_layer'] is not None]
            
            for layer in stab_layers:
                stabilization_data.append({
                    'Model': analysis['model_info']['model_name'],
                    'Question_Type': analysis['model_info']['question_type'],
                    'Stabilization_Layer': layer,
                    'Key': key
                })
        
        if stabilization_data:
            stab_df = pd.DataFrame(stabilization_data)
            sns.boxplot(data=stab_df, x='Question_Type', y='Stabilization_Layer', hue='Model', ax=ax1)
            ax1.set_title('Distribution of Stabilization Layers')
            ax1.set_ylabel('Layer Number')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: Confidence evolution across layers
        ax2 = axes[0, 1]
        for key, analysis in self.analysis_results.items():
            layer_analysis = analysis['layer_analysis']
            
            # Average confidence by layer
            layer_confidences = {}
            for item in layer_analysis:
                for i, (layer_num, conf) in enumerate(zip(item['layer_numbers'], item['confidences_by_layer'])):
                    if layer_num not in layer_confidences:
                        layer_confidences[layer_num] = []
                    layer_confidences[layer_num].append(conf)
            
            if layer_confidences:
                layers = sorted(layer_confidences.keys())
                mean_confs = [np.mean(layer_confidences[layer]) for layer in layers]
                ax2.plot(layers, mean_confs, marker='o', label=f"{analysis['model_info']['model_name']}_{analysis['model_info']['question_type']}")
        
        ax2.set_title('Average Confidence Across Layers')
        ax2.set_xlabel('Layer Number')
        ax2.set_ylabel('Average Confidence')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Comparison of plain vs opinion (if available)
        ax3 = axes[1, 0]
        comparison_data = self.compare_plain_vs_opinion()
        
        if comparison_data:
            models = list(comparison_data.keys())
            plain_means = [comparison_data[model]['plain_mean_stab'] for model in models]
            opinion_means = [comparison_data[model]['opinion_mean_stab'] for model in models]
            
            x = np.arange(len(models))
            width = 0.35
            
            ax3.bar(x - width/2, plain_means, width, label='Plain Questions', alpha=0.8)
            ax3.bar(x + width/2, opinion_means, width, label='Opinion Questions', alpha=0.8)
            
            ax3.set_title('Mean Stabilization Layer: Plain vs Opinion')
            ax3.set_xlabel('Model')
            ax3.set_ylabel('Mean Stabilization Layer')
            ax3.set_xticks(x)
            ax3.set_xticklabels(models, rotation=45, ha='right')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No plain vs opinion\ncomparison available', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Plain vs Opinion Comparison')
        
        # Plot 4: Sycophancy emergence (if available)
        ax4 = axes[1, 1]
        sycophancy_data = []
        
        for key, analysis in self.analysis_results.items():
            if 'sycophancy_analysis' in analysis:
                syc_analysis = analysis['sycophancy_analysis']
                syc_rates = []
                
                # Calculate sycophancy rate by layer
                max_layer = max([max(x['layer_numbers']) for x in syc_analysis if x['layer_numbers']])
                
                for layer in range(1, max_layer + 1):
                    layer_sycophancy = []
                    for item in syc_analysis:
                        if layer <= len(item['opinion_alignment_by_layer']):
                            layer_sycophancy.append(item['opinion_alignment_by_layer'][layer-1])
                    
                    if layer_sycophancy:
                        syc_rate = np.mean(layer_sycophancy)
                        sycophancy_data.append({
                            'Layer': layer,
                            'Sycophancy_Rate': syc_rate,
                            'Model': analysis['model_info']['model_name']
                        })
        
        if sycophancy_data:
            syc_df = pd.DataFrame(sycophancy_data)
            for model in syc_df['Model'].unique():
                model_data = syc_df[syc_df['Model'] == model]
                ax4.plot(model_data['Layer'], model_data['Sycophancy_Rate'], 
                        marker='o', label=model, linewidth=2)
            
            ax4.set_title('Sycophancy Rate Across Layers')
            ax4.set_xlabel('Layer Number')
            ax4.set_ylabel('Sycophancy Rate')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No sycophancy analysis\navailable', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Sycophancy Emergence')
        
        plt.tight_layout()
        plt.savefig('early_decoding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional detailed plots
        self.create_detailed_plots()
    
    def create_detailed_plots(self):
        """Create additional detailed analysis plots."""
        
        # Layer-wise prediction consistency heatmap
        fig, axes = plt.subplots(1, len(self.analysis_results), figsize=(5*len(self.analysis_results), 6))
        if len(self.analysis_results) == 1:
            axes = [axes]
        
        for idx, (key, analysis) in enumerate(self.analysis_results.items()):
            layer_analysis = analysis['layer_analysis']
            
            # Create consistency matrix
            all_layers = set()
            for item in layer_analysis:
                all_layers.update(item['layer_numbers'])
            
            max_layer = max(all_layers) if all_layers else 20
            
            consistency_matrix = np.full((len(layer_analysis), max_layer), np.nan)
            
            for i, item in enumerate(layer_analysis):
                final_pred = item['final_prediction']
                for j, (layer_num, pred) in enumerate(zip(item['layer_numbers'], item['predictions_by_layer'])):
                    if layer_num <= max_layer:
                        consistency_matrix[i, layer_num-1] = 1 if pred == final_pred else 0
            
            # Plot heatmap
            if idx < len(axes):
                im = axes[idx].imshow(consistency_matrix, cmap='RdYlBu', aspect='auto', vmin=0, vmax=1)
                axes[idx].set_title(f'{key}\nPrediction Consistency')
                axes[idx].set_xlabel('Layer Number')
                axes[idx].set_ylabel('Question Index')
                
                # Add colorbar
                plt.colorbar(im, ax=axes[idx], label='Consistent with Final Prediction')
        
        plt.tight_layout()
        plt.savefig('prediction_consistency_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report."""
        print("\n" + "="*60)
        print("EARLY DECODING ANALYSIS SUMMARY REPORT")
        print("="*60)
        
        for key, analysis in self.analysis_results.items():
            print(f"\n{key.upper()}")
            print("-" * len(key))
            
            layer_analysis = analysis['layer_analysis']
            model_info = analysis['model_info']
            
            # Basic statistics
            total_questions = len(layer_analysis)
            valid_predictions = len([x for x in layer_analysis if x['final_prediction'] is not None])
            
            print(f"Total questions: {total_questions}")
            print(f"Valid predictions: {valid_predictions}")
            
            # Stabilization statistics
            stab_layers = [x['stabilization_layer'] for x in layer_analysis if x['stabilization_layer'] is not None]
            if stab_layers:
                print(f"Questions that stabilize: {len(stab_layers)} ({len(stab_layers)/total_questions*100:.1f}%)")
                print(f"Mean stabilization layer: {np.mean(stab_layers):.2f}")
                print(f"Median stabilization layer: {np.median(stab_layers):.2f}")
                print(f"Std deviation: {np.std(stab_layers):.2f}")
                print(f"Range: {min(stab_layers)} - {max(stab_layers)}")
            
            # Accuracy if available
            correct_predictions = [x for x in layer_analysis if x['is_correct'] is True]
            if correct_predictions:
                accuracy = len(correct_predictions) / len([x for x in layer_analysis if x['is_correct'] is not None])
                print(f"Accuracy: {accuracy:.3f}")
            
            # Sycophancy analysis if available
            if 'sycophancy_analysis' in analysis:
                syc_analysis = analysis['sycophancy_analysis']
                sycophantic = [x for x in syc_analysis if x['is_sycophantic']]
                print(f"Sycophantic responses: {len(sycophantic)}/{len(syc_analysis)} ({len(sycophantic)/len(syc_analysis)*100:.1f}%)")
                
                syc_start_layers = [x['sycophancy_start_layer'] for x in sycophantic if x['sycophancy_start_layer'] is not None]
                if syc_start_layers:
                    print(f"Mean sycophancy emergence layer: {np.mean(syc_start_layers):.2f}")
        
        # Cross-model comparisons
        comparison_results = self.compare_plain_vs_opinion()
        if comparison_results:
            print(f"\nCROSS-MODEL COMPARISONS")
            print("-" * 25)
            
            for model, comp_data in comparison_results.items():
                print(f"\n{model}:")
                print(f"  Plain questions - Mean stabilization: {comp_data['plain_mean_stab']:.2f}")
                print(f"  Opinion questions - Mean stabilization: {comp_data['opinion_mean_stab']:.2f}")
                print(f"  Difference: {comp_data['opinion_mean_stab'] - comp_data['plain_mean_stab']:.2f}")
                print(f"  Effect size: {comp_data['effect_size']:.3f}")
                print(f"  Statistical significance (p-value): {comp_data['p_value']:.4f}")
        
        print("\n" + "="*60)
        print("Analysis complete! Check the generated plots for visual insights.")
        print("="*60)

def main():
    # Initialize analyzer
    analyzer = EarlyDecodingAnalysis()
    
    # Load all result files
    analyzer.load_results()
    
    if not analyzer.results:
        print("No result files found. Please check the directory structure.")
        return
    
    # Perform analyses
    analyzer.analyze_early_decoding_patterns()
    analyzer.analyze_sycophancy_emergence()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()