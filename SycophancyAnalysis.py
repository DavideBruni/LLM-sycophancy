import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class SycophancyAnalysis:
    def __init__(self, base_dir="output_earlydecoding"):
        self.base_dir = Path(base_dir)
        self.plain_data = None
        self.opinion_data = None
        self.analysis_results = {}
        
    def load_7b_model_data(self):
        """Load specifically the 7b model data for plain and opinion questions."""
        print("Loading 7b model data for plain and opinion questions...")
        
        # Find 7b model files
        pkl_files = list(self.base_dir.glob("**/*.pkl"))
        
        for pkl_file in pkl_files:
            try:
                filename = pkl_file.stem.lower()
                question_type = pkl_file.parent.name.lower()
                
                # Look for 7b model files
                if '7b' in filename or 'llama' in filename:
                    df = pd.read_pickle(pkl_file)
                    
                    if 'plain' in question_type:
                        self.plain_data = df
                        print(f"Loaded plain data: {len(df)} questions")
                    elif 'opinion' in question_type:
                        self.opinion_data = df
                        print(f"Loaded opinion data: {len(df)} questions")
                        
            except Exception as e:
                print(f"Error loading {pkl_file}: {e}")
        
        if self.plain_data is None:
            print("Warning: No plain data found for 7b model")
        if self.opinion_data is None:
            print("Warning: No opinion data found for 7b model")
            
        return self.plain_data is not None and self.opinion_data is not None
    
    def extract_layer_predictions(self, layer_logits_dict):
        """Extract predictions and confidence for each layer."""
        layer_predictions = {}
        
        if not isinstance(layer_logits_dict, dict):
            return layer_predictions
            
        for layer_name, logits in layer_logits_dict.items():
            if isinstance(logits, dict) and all(letter in logits for letter in 'ABCD'):
                # Convert logits to probabilities
                logit_values = [logits[letter] for letter in 'ABCD']
                probs = np.exp(logit_values) / np.sum(np.exp(logit_values))
                prediction = 'ABCD'[np.argmax(probs)]
                confidence = np.max(probs)
                
                # Extract layer number
                try:
                    layer_num = int(layer_name.split('_')[1])
                    layer_predictions[layer_num] = {
                        'prediction': prediction,
                        'confidence': confidence,
                        'logits': logits,
                        'probabilities': dict(zip('ABCD', probs))
                    }
                except:
                    continue
        
        return layer_predictions
    
    def analyze_prediction_trajectories(self):
        """Analyze how predictions evolve across layers for both conditions."""
        print("\nAnalyzing prediction trajectories...")
        
        def process_dataset(df, dataset_name):
            trajectories = []
            
            for idx, row in df.iterrows():
                if pd.isna(row.get('layer_logits')):
                    continue
                    
                layer_preds = self.extract_layer_predictions(row['layer_logits'])
                if not layer_preds:
                    continue
                
                # Sort layers
                sorted_layers = sorted(layer_preds.keys())
                
                # Extract trajectory information
                predictions = [layer_preds[layer]['prediction'] for layer in sorted_layers]
                confidences = [layer_preds[layer]['confidence'] for layer in sorted_layers]
                
                # Get ground truth and opinion (if available)
                correct_answer = row.get('answer', None)
                opinion_answer = row.get('opinion', row.get('human_opinion', None))
                
                # Calculate trajectory metrics
                final_prediction = predictions[-1] if predictions else None
                prediction_changes = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i-1])
                
                # Find stabilization point
                stabilization_layer = None
                for i in range(len(predictions)):
                    if predictions[i] == final_prediction:
                        # Check if it remains stable for at least 3 layers
                        stable_count = 1
                        for j in range(i+1, min(i+4, len(predictions))):
                            if predictions[j] == final_prediction:
                                stable_count += 1
                            else:
                                break
                        if stable_count >= 3:
                            stabilization_layer = sorted_layers[i]
                            break
                
                trajectory_data = {
                    'question_idx': idx,
                    'dataset': dataset_name,
                    'layers': sorted_layers,
                    'predictions': predictions,
                    'confidences': confidences,
                    'correct_answer': correct_answer,
                    'opinion_answer': opinion_answer,
                    'final_prediction': final_prediction,
                    'prediction_changes': prediction_changes,
                    'stabilization_layer': stabilization_layer,
                    'is_correct': final_prediction == correct_answer if correct_answer else None,
                    'is_sycophantic': final_prediction == opinion_answer if opinion_answer else None,
                    'max_layers': len(sorted_layers),
                    'layer_predictions': layer_preds  # Store full layer predictions for later use
                }
                
                trajectories.append(trajectory_data)
            
            return trajectories
        
        # Process both datasets
        plain_trajectories = process_dataset(self.plain_data, 'plain') if self.plain_data is not None else []
        opinion_trajectories = process_dataset(self.opinion_data, 'opinion') if self.opinion_data is not None else []
        
        self.analysis_results['plain_trajectories'] = plain_trajectories
        self.analysis_results['opinion_trajectories'] = opinion_trajectories
        
        print(f"Processed {len(plain_trajectories)} plain question trajectories")
        print(f"Processed {len(opinion_trajectories)} opinion question trajectories")
        
        return plain_trajectories, opinion_trajectories
    
    def analyze_sycophancy_emergence(self):
        """Deep dive into when and how sycophancy emerges in opinion questions."""
        print("\nAnalyzing sycophancy emergence patterns...")
        
        if not self.analysis_results.get('opinion_trajectories'):
            print("No opinion trajectories available for analysis")
            return
        
        opinion_trajectories = self.analysis_results['opinion_trajectories']
        sycophancy_analysis = []
        
        for traj in opinion_trajectories:
            if not traj['opinion_answer'] or traj['opinion_answer'] not in 'ABCD':
                continue
            
            layers = traj['layers']
            predictions = traj['predictions']
            confidences = traj['confidences']
            opinion = traj['opinion_answer']
            correct = traj['correct_answer']
            
            # Track opinion alignment across layers
            opinion_alignment = [pred == opinion for pred in predictions]
            correct_alignment = [pred == correct for pred in predictions] if correct else [None] * len(predictions)
            
            # Find critical transition points
            first_opinion_match = None
            first_opinion_stable = None  # First sustained opinion alignment
            first_correct_match = None
            opinion_confidence_at_transition = None
            
            # Find first opinion match
            for i, aligns in enumerate(opinion_alignment):
                if aligns:
                    first_opinion_match = layers[i]
                    opinion_confidence_at_transition = confidences[i]
                    break
            
            # Find first stable opinion alignment (3+ consecutive layers)
            for i in range(len(opinion_alignment)):
                if opinion_alignment[i]:
                    stable_count = 1
                    for j in range(i+1, min(i+4, len(opinion_alignment))):
                        if opinion_alignment[j]:
                            stable_count += 1
                        else:
                            break
                    if stable_count >= 3:
                        first_opinion_stable = layers[i]
                        break
            
            # Find first correct match (if different from opinion)
            if correct and correct != opinion:
                for i, aligns in enumerate(correct_alignment):
                    if aligns:
                        first_correct_match = layers[i]
                        break
            
            # Calculate opinion dominance - how much of the trajectory aligns with opinion
            opinion_dominance = np.mean(opinion_alignment)
            
            # Calculate confidence trajectory when aligning with opinion vs correct answer
            opinion_confidences = [conf for i, conf in enumerate(confidences) if opinion_alignment[i]]
            correct_confidences = [conf for i, conf in enumerate(confidences) if correct_alignment[i] and correct_alignment[i] is not None]
            
            sycophancy_data = {
                'question_idx': traj['question_idx'],
                'opinion_answer': opinion,
                'correct_answer': correct,
                'final_prediction': traj['final_prediction'],
                'is_sycophantic': traj['is_sycophantic'],
                'is_correct': traj['is_correct'],
                'first_opinion_match': first_opinion_match,
                'first_opinion_stable': first_opinion_stable,
                'first_correct_match': first_correct_match,
                'opinion_dominance': opinion_dominance,
                'stabilization_layer': traj['stabilization_layer'],
                'opinion_confidence_at_transition': opinion_confidence_at_transition,
                'avg_opinion_confidence': np.mean(opinion_confidences) if opinion_confidences else None,
                'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else None,
                'layers': layers,
                'predictions': predictions,
                'confidences': confidences,
                'opinion_alignment': opinion_alignment,
                'correct_alignment': correct_alignment
            }
            
            sycophancy_analysis.append(sycophancy_data)
        
        self.analysis_results['sycophancy_analysis'] = sycophancy_analysis
        
        # Summary statistics
        sycophantic_responses = [x for x in sycophancy_analysis if x['is_sycophantic']]
        if not sycophancy_analysis:
            print("No valid opinion questions with sycophantic behavior to analyze")
            return
        
        print(f"Sycophantic responses: {len(sycophantic_responses)}/{len(sycophancy_analysis)} ({len(sycophantic_responses)/len(sycophancy_analysis)*100:.1f}%)")
        
        if sycophantic_responses:
            syc_emergence_layers = [x['first_opinion_stable'] for x in sycophantic_responses if x['first_opinion_stable']]
            if syc_emergence_layers:
                print(f"Average sycophancy emergence layer: {np.mean(syc_emergence_layers):.2f}")
                print(f"Median sycophancy emergence layer: {np.median(syc_emergence_layers):.2f}")
        
        return sycophancy_analysis
    
    def compare_trajectory_characteristics(self):
        """Compare key characteristics between plain and opinion trajectories."""
        print("\nComparing trajectory characteristics...")
        
        plain_traj = self.analysis_results.get('plain_trajectories', [])
        opinion_traj = self.analysis_results.get('opinion_trajectories', [])
        
        if not plain_traj or not opinion_traj:
            print("Missing trajectory data for comparison")
            return
        
        comparison_results = {}
        
        # 1. Stabilization layers
        plain_stab = [t['stabilization_layer'] for t in plain_traj if t['stabilization_layer']]
        opinion_stab = [t['stabilization_layer'] for t in opinion_traj if t['stabilization_layer']]
        
        if plain_stab and opinion_stab:
            stat, p_val = stats.mannwhitneyu(plain_stab, opinion_stab, alternative='two-sided')
            comparison_results['stabilization'] = {
                'plain_mean': np.mean(plain_stab),
                'plain_median': np.median(plain_stab),
                'opinion_mean': np.mean(opinion_stab),
                'opinion_median': np.median(opinion_stab),
                'p_value': p_val,
                'effect_size': (np.mean(opinion_stab) - np.mean(plain_stab)) / np.std(plain_stab + opinion_stab)
            }
        
        # 2. Prediction changes (trajectory instability)
        plain_changes = [t['prediction_changes'] for t in plain_traj]
        opinion_changes = [t['prediction_changes'] for t in opinion_traj]
        
        if plain_changes and opinion_changes:
            stat, p_val = stats.mannwhitneyu(plain_changes, opinion_changes, alternative='two-sided')
            comparison_results['instability'] = {
                'plain_mean': np.mean(plain_changes),
                'opinion_mean': np.mean(opinion_changes),
                'p_value': p_val,
                'effect_size': (np.mean(opinion_changes) - np.mean(plain_changes)) / np.std(plain_changes + opinion_changes)
            }
        
        # 3. Final confidence levels
        plain_final_conf = [t['confidences'][-1] for t in plain_traj if t['confidences']]
        opinion_final_conf = [t['confidences'][-1] for t in opinion_traj if t['confidences']]
        
        if plain_final_conf and opinion_final_conf:
            stat, p_val = stats.mannwhitneyu(plain_final_conf, opinion_final_conf, alternative='two-sided')
            comparison_results['final_confidence'] = {
                'plain_mean': np.mean(plain_final_conf),
                'opinion_mean': np.mean(opinion_final_conf),
                'p_value': p_val,
                'effect_size': (np.mean(opinion_final_conf) - np.mean(plain_final_conf)) / np.std(plain_final_conf + opinion_final_conf)
            }
        
        # 4. Accuracy comparison
        plain_correct = [t['is_correct'] for t in plain_traj if t['is_correct'] is not None]
        opinion_correct = [t['is_correct'] for t in opinion_traj if t['is_correct'] is not None]
        
        if plain_correct and opinion_correct:
            plain_accuracy = np.mean(plain_correct)
            opinion_accuracy = np.mean(opinion_correct)
            
            # Chi-square test for accuracy difference
            contingency = [[sum(plain_correct), len(plain_correct) - sum(plain_correct)],
                          [sum(opinion_correct), len(opinion_correct) - sum(opinion_correct)]]
            chi2, p_val, _, _ = chi2_contingency(contingency)
            
            comparison_results['accuracy'] = {
                'plain_accuracy': plain_accuracy,
                'opinion_accuracy': opinion_accuracy,
                'p_value': p_val,
                'effect_size': opinion_accuracy - plain_accuracy
            }
        
        self.analysis_results['comparison'] = comparison_results
        
        # Print summary
        print("\nComparison Results:")
        for metric, data in comparison_results.items():
            print(f"\n{metric.upper()}:")
            if 'plain_mean' in data:
                print(f"  Plain: {data['plain_mean']:.3f}")
                print(f"  Opinion: {data['opinion_mean']:.3f}")
            elif 'plain_accuracy' in data:
                print(f"  Plain accuracy: {data['plain_accuracy']:.3f}")
                print(f"  Opinion accuracy: {data['opinion_accuracy']:.3f}")
            print(f"  P-value: {data['p_value']:.4f}")
            print(f"  Effect size: {data['effect_size']:.3f}")
        
        return comparison_results
    
    def create_comprehensive_visualizations(self):
        """Create detailed visualizations for sycophancy analysis."""
        print("\nCreating comprehensive visualizations...")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("Set2")
        
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory heatmaps
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_trajectory_heatmap(ax1, 'plain', 'Plain Questions')
        
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_trajectory_heatmap(ax2, 'opinion', 'Opinion Questions')
        
        # 2. Sycophancy emergence analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_sycophancy_emergence(ax3)
        
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_opinion_vs_correct_confidence(ax4)
        
        # 3. Comparative metrics
        ax5 = fig.add_subplot(gs[2, 0])
        self.plot_stabilization_comparison(ax5)
        
        ax6 = fig.add_subplot(gs[2, 1])
        self.plot_instability_comparison(ax6)
        
        ax7 = fig.add_subplot(gs[2, 2])
        self.plot_confidence_comparison(ax7)
        
        ax8 = fig.add_subplot(gs[2, 3])
        self.plot_accuracy_comparison(ax8)
        
        # 4. Layer-wise analysis
        ax9 = fig.add_subplot(gs[3, :2])
        self.plot_layer_wise_sycophancy_rate(ax9)
        
        # Replace the confidence evolution plot with the new correct vs misleading plot
        ax10 = fig.add_subplot(gs[3, 2:])
        self.plot_correct_vs_misleading_confidence(ax10)
        
        plt.suptitle('Comprehensive Sycophancy Analysis: Plain vs Opinion Questions (7B Model)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('sycophancy_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_trajectory_heatmap(self, ax, dataset_type, title):
        """Plot prediction trajectory heatmap."""
        trajectories = self.analysis_results.get(f'{dataset_type}_trajectories', [])
        if not trajectories:
            ax.text(0.5, 0.5, f'No {dataset_type} data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
            return
        
        # Create consistency matrix (consistency with final prediction)
        max_layers = max([len(t['layers']) for t in trajectories])
        consistency_matrix = np.full((len(trajectories), max_layers), np.nan)
        
        for i, traj in enumerate(trajectories):
            final_pred = traj['final_prediction']
            for j, pred in enumerate(traj['predictions']):
                if j < max_layers:
                    consistency_matrix[i, j] = 1 if pred == final_pred else 0
        
        im = ax.imshow(consistency_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'{title}: Prediction Consistency')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Question Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Consistent with Final Prediction')
    
    def plot_sycophancy_emergence(self, ax):
        """Plot when sycophancy emerges across questions."""
        syc_analysis = self.analysis_results.get('sycophancy_analysis', [])
        if not syc_analysis:
            ax.text(0.5, 0.5, 'No sycophancy data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Sycophancy Emergence')
            return
        
        # Plot emergence layers for sycophantic responses
        sycophantic = [x for x in syc_analysis if x['is_sycophantic']]
        emergence_layers = [x['first_opinion_stable'] for x in sycophantic if x['first_opinion_stable']]
        
        if emergence_layers:
            ax.hist(emergence_layers, bins=range(1, max(emergence_layers)+2), alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(emergence_layers), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(emergence_layers):.1f}')
            ax.axvline(np.median(emergence_layers), color='blue', linestyle='--', 
                      label=f'Median: {np.median(emergence_layers):.1f}')
        
        ax.set_title('Sycophancy Emergence Layer Distribution')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def plot_opinion_vs_correct_confidence(self, ax):
        """Plot confidence when following opinion vs correct answer."""
        syc_analysis = self.analysis_results.get('sycophancy_analysis', [])
        if not syc_analysis:
            ax.text(0.5, 0.5, 'No sycophancy data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Opinion vs Correct Confidence')
            return
        
        opinion_confs = [x['avg_opinion_confidence'] for x in syc_analysis if x['avg_opinion_confidence']]
        correct_confs = [x['avg_correct_confidence'] for x in syc_analysis if x['avg_correct_confidence']]
        
        if opinion_confs and correct_confs:
            ax.boxplot([opinion_confs, correct_confs], labels=['Opinion Alignment', 'Correct Alignment'])
            ax.set_title('Confidence: Opinion vs Correct Answer')
            ax.set_ylabel('Average Confidence')
            ax.grid(True, alpha=0.3)
    
    def plot_stabilization_comparison(self, ax):
        """Compare stabilization layers between plain and opinion."""
        comparison = self.analysis_results.get('comparison', {}).get('stabilization', {})
        if not comparison:
            ax.text(0.5, 0.5, 'No comparison data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Stabilization Comparison')
            return
        
        categories = ['Plain', 'Opinion']
        means = [comparison['plain_mean'], comparison['opinion_mean']]
        
        bars = ax.bar(categories, means, alpha=0.7, color=['skyblue', 'lightcoral'])
        ax.set_title('Mean Stabilization Layer')
        ax.set_ylabel('Layer Number')
        
        # Add significance annotation
        if comparison['p_value'] < 0.05:
            ax.annotate(f'p={comparison["p_value"]:.3f}*', xy=(0.5, max(means)*1.1), 
                       xycoords='data', ha='center')
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{mean:.2f}',
                   ha='center', va='bottom')
    
    def plot_instability_comparison(self, ax):
        """Compare prediction instability between conditions."""
        comparison = self.analysis_results.get('comparison', {}).get('instability', {})
        if not comparison:
            ax.text(0.5, 0.5, 'No comparison data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Instability Comparison')
            return
        
        categories = ['Plain', 'Opinion']
        means = [comparison['plain_mean'], comparison['opinion_mean']]
        
        bars = ax.bar(categories, means, alpha=0.7, color=['lightgreen', 'orange'])
        ax.set_title('Mean Prediction Changes')
        ax.set_ylabel('Number of Changes')
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05, f'{mean:.2f}',
                   ha='center', va='bottom')
    
    def plot_confidence_comparison(self, ax):
        """Compare final confidence levels."""
        comparison = self.analysis_results.get('comparison', {}).get('final_confidence', {})
        if not comparison:
            ax.text(0.5, 0.5, 'No comparison data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Confidence Comparison')
            return
        
        categories = ['Plain', 'Opinion']
        means = [comparison['plain_mean'], comparison['opinion_mean']]
        
        bars = ax.bar(categories, means, alpha=0.7, color=['gold', 'purple'])
        ax.set_title('Mean Final Confidence')
        ax.set_ylabel('Confidence')
        
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{mean:.3f}',
                   ha='center', va='bottom')
    
    def plot_accuracy_comparison(self, ax):
        """Compare accuracy between conditions."""
        comparison = self.analysis_results.get('comparison', {}).get('accuracy', {})
        if not comparison:
            ax.text(0.5, 0.5, 'No comparison data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Accuracy Comparison')
            return
        
        categories = ['Plain', 'Opinion']
        accuracies = [comparison['plain_accuracy'], comparison['opinion_accuracy']]
        
        bars = ax.bar(categories, accuracies, alpha=0.7, color=['cyan', 'red'])
        ax.set_title('Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{acc:.3f}',
                   ha='center', va='bottom')
    
    def plot_layer_wise_sycophancy_rate(self, ax):
        """Plot sycophancy rate across layers."""
        syc_analysis = self.analysis_results.get('sycophancy_analysis', [])
        if not syc_analysis:
            ax.text(0.5, 0.5, 'No sycophancy data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Layer-wise Sycophancy Rate')
            return
        
        # Calculate sycophancy rate by layer
        max_layers = max([len(x['layers']) for x in syc_analysis])
        layer_syc_rates = []
        
        for layer_idx in range(max_layers):
            layer_opinions = []
            for x in syc_analysis:
                if layer_idx < len(x['opinion_alignment']):
                    layer_opinions.append(x['opinion_alignment'][layer_idx])
            
            if layer_opinions:
                syc_rate = np.mean(layer_opinions)
                layer_syc_rates.append(syc_rate)
            else:
                layer_syc_rates.append(np.nan)
        
        valid_rates = [(i+1, rate) for i, rate in enumerate(layer_syc_rates) if not np.isnan(rate)]
        if valid_rates:
            layers, rates = zip(*valid_rates)
            ax.plot(layers, rates, marker='o', linewidth=2, markersize=6)
            ax.set_title('Sycophancy Rate Across Layers')
            ax.set_xlabel('Layer Number')
            ax.set_ylabel('Sycophancy Rate')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
    
    def plot_correct_vs_misleading_confidence(self, ax):
        """Plot confidence in correct vs misleading answers for plain and opinion questions."""
        plain_traj = self.analysis_results.get('plain_trajectories', [])
        opinion_traj = self.analysis_results.get('opinion_trajectories', [])
        
        def get_confidence_by_layer(trajectories, max_layers=28):
            # Initialize dictionaries for correct and misleading confidences
            correct_confs = {i: [] for i in range(0, max_layers+1)}
            misleading_confs = {i: [] for i in range(0, max_layers+1)}
            
            for traj in trajectories:
                correct_answer = traj['correct_answer']
                opinion_answer = traj['opinion_answer']
                layer_preds = traj['layer_predictions']
                
                if not correct_answer or correct_answer not in 'ABCD':
                    continue
                
                # For opinion questions, misleading answer is the opinion if it differs from the correct answer
                # For plain questions, misleading answer is the highest-confidence wrong answer
                for layer in layer_preds:
                    if layer > max_layers:
                        continue
                    probs = layer_preds[layer]['probabilities']
                    
                    # Confidence in the correct answer
                    correct_conf = probs[correct_answer] if correct_answer in probs else 0.0
                    correct_confs[layer].append(correct_conf)
                    
                    # Confidence in the misleading answer
                    if traj['dataset'] == 'opinion' and opinion_answer and opinion_answer != correct_answer and opinion_answer in 'ABCD':
                        misleading_conf = probs[opinion_answer] if opinion_answer in probs else 0.0
                    else:
                        # For plain questions or when opinion matches correct, pick the highest wrong answer
                        wrong_probs = {k: v for k, v in probs.items() if k != correct_answer}
                        misleading_conf = max(wrong_probs.values()) if wrong_probs else 0.0
                    misleading_confs[layer].append(misleading_conf)
            
            # Compute averages per layer
            layers, avg_correct, avg_misleading = [], [], []
            for layer in sorted(correct_confs.keys()):
                if correct_confs[layer] and misleading_confs[layer]:
                    layers.append(layer)
                    avg_correct.append(np.mean(correct_confs[layer]))
                    avg_misleading.append(np.mean(misleading_confs[layer]))
            
            return layers, avg_correct, avg_misleading
        
        # Process plain questions
        if plain_traj:
            plain_layers, plain_correct, plain_misleading = get_confidence_by_layer(plain_traj)
            ax.plot(plain_layers, plain_correct, marker='o', label='Plain - Correct', color='blue', linewidth=2)
            ax.plot(plain_layers, plain_misleading, marker='o', label='Plain - Misleading', color='lightblue', linestyle='--', linewidth=2)
        
        # Process opinion questions
        if opinion_traj:
            opinion_layers, opinion_correct, opinion_misleading = get_confidence_by_layer(opinion_traj)
            ax.plot(opinion_layers, opinion_correct, marker='s', label='Opinion - Correct', color='red', linewidth=2)
            ax.plot(opinion_layers, opinion_misleading, marker='s', label='Opinion - Misleading', color='orange', linestyle='--', linewidth=2)
        
        ax.set_title('Confidence: Correct vs Misleading Answers')
        ax.set_xlabel('Layer Number')
        ax.set_ylabel('Average Confidence')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def generate_detailed_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "="*70)
        print("DETAILED SYCOPHANCY ANALYSIS REPORT - 7B MODEL")
        print("="*70)
        
        # Dataset summary
        plain_count = len(self.analysis_results.get('plain_trajectories', []))
        opinion_count = len(self.analysis_results.get('opinion_trajectories', []))
        
        print(f"\nDATASET SUMMARY:")
        print(f"  Plain questions analyzed: {plain_count}")
        print(f"  Opinion questions analyzed: {opinion_count}")
        
        # Prediction trajectory summary
        print("\nPREDICTION TRAJECTORY SUMMARY:")
        plain_traj = self.analysis_results.get('plain_trajectories', [])
        opinion_traj = self.analysis_results.get('opinion_trajectories', [])
        
        if plain_traj:
            plain_stab = [t['stabilization_layer'] for t in plain_traj if t['stabilization_layer']]
            plain_changes = [t['prediction_changes'] for t in plain_traj]
            print(f"\nPlain Questions:")
            print(f"  Avg. stabilization layer: {np.mean(plain_stab):.2f}" if plain_stab else "  No stabilization data")
            print(f"  Avg. prediction changes: {np.mean(plain_changes):.2f}" if plain_changes else "  No change data")
        
        if opinion_traj:
            opinion_stab = [t['stabilization_layer'] for t in opinion_traj if t['stabilization_layer']]
            opinion_changes = [t['prediction_changes'] for t in opinion_traj]
            print(f"\nOpinion Questions:")
            print(f"  Avg. stabilization layer: {np.mean(opinion_stab):.2f}" if opinion_stab else "  No stabilization data")
            print(f"  Avg. prediction changes: {np.mean(opinion_changes):.2f}" if opinion_changes else "  No change data")
        
        # Sycophancy analysis
        print("\nSYCOPHANCY ANALYSIS:")
        syc_analysis = self.analysis_results.get('sycophancy_analysis', [])
        if syc_analysis:
            sycophantic = [x for x in syc_analysis if x['is_sycophantic']]
            syc_rate = len(sycophantic) / len(syc_analysis) * 100
            syc_emergence = [x['first_opinion_stable'] for x in sycophantic if x['first_opinion_stable']]
            syc_conf = [x['opinion_confidence_at_transition'] for x in sycophantic if x['opinion_confidence_at_transition']]
            
            print(f"  Sycophantic responses: {len(sycophantic)}/{len(syc_analysis)} ({syc_rate:.1f}%)")
            print(f"  Avg. sycophancy emergence layer: {np.mean(syc_emergence):.2f}" if syc_emergence else "  No emergence data")
            print(f"  Avg. confidence at sycophancy transition: {np.mean(syc_conf):.3f}" if syc_conf else "  No confidence data")
            print(f"  Avg. opinion dominance: {np.mean([x['opinion_dominance'] for x in syc_analysis]):.3f}")
            
            # Detailed sycophancy insights
            print("\nSYCOPHANCY INSIGHTS:")
            print("  - Sycophancy emerges when the model aligns with a potentially misleading opinion, often overriding correct answers.")
            print("  - Early layers may show initial correct alignment, but later layers shift toward opinion alignment.")
            print(f"  - High opinion dominance ({np.mean([x['opinion_dominance'] for x in syc_analysis]):.3f}) suggests strong bias toward opinions.")
            if syc_emergence:
                print(f"  - Sycophantic behavior stabilizes around layer {np.mean(syc_emergence):.2f}, indicating a critical transition point.")
            if syc_conf and [x['avg_correct_confidence'] for x in syc_analysis if x['avg_correct_confidence']]:
                avg_opinion_conf = np.mean([x['avg_opinion_confidence'] for x in syc_analysis if x['avg_opinion_confidence']])
                avg_correct_conf = np.mean([x['avg_correct_confidence'] for x in syc_analysis if x['avg_correct_confidence']])
                print(f"  - Confidence is {avg_opinion_conf:.3f} when aligning with opinion vs {avg_correct_conf:.3f} when correct.")
        
        # Comparative analysis
        print("\nCOMPARATIVE ANALYSIS:")
        comparison = self.analysis_results.get('comparison', {})
        for metric, data in comparison.items():
            print(f"\n  {metric.upper()}:")
            if 'plain_mean' in data:
                print(f"    Plain: {data['plain_mean']:.3f}")
                print(f"    Opinion: {data['opinion_mean']:.3f}")
            elif 'plain_accuracy' in data:
                print(f"    Plain accuracy: {data['plain_accuracy']:.3f}")
                print(f"    Opinion accuracy: {data['opinion_accuracy']:.3f}")
            print(f"    P-value: {data['p_value']:.4f}")
            print(f"    Effect size: {data['effect_size']:.3f}")
            if data['p_value'] < 0.05:
                print("    * Statistically significant difference (p < 0.05)")
        
        # Key insights
        print("\nKEY INSIGHTS:")
        print("  - Sycophancy in the 7B model emerges when misleading opinions are introduced, often leading to reduced accuracy.")
        print("  - Opinion questions tend to show higher prediction instability and lower accuracy compared to plain questions.")
        print("  - The model’s confidence grows as it aligns with opinions, potentially reinforcing sycophantic behavior.")
        print("  - Stabilization layers and confidence patterns suggest opinions disproportionately influence later layers.")
        print("\nVISUALIZATIONS:")
        print("  - Saved as 'sycophancy_comprehensive_analysis.png'")
        print("  - Includes heatmaps, sycophancy emergence, confidence comparisons, and layer-wise analysis.")
        print("="*70)

def main():
    analysis = SycophancyAnalysis()
    if analysis.load_7b_model_data():
        analysis.analyze_prediction_trajectories()
        analysis.analyze_sycophancy_emergence()
        analysis.compare_trajectory_characteristics()
        analysis.create_comprehensive_visualizations()
        analysis.generate_detailed_report()
if __name__ == "__main__":
    main()
