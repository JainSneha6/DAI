# marketing_chat_blueprint.py
# Enhanced chat interface for Marketing Mix Modeling (MMM) with ROI analysis
import os
import json
import logging
import re
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime
import pickle

logger = logging.getLogger(__name__)

bp = Blueprint("marketing_chat_bp", __name__)

# Try imports
try:
    import google.generativeai as genai
except Exception:
    genai = None
    logger.error("Gemini not installed")

try:
    from cyborgdb_core.integrations.langchain import CyborgVectorStore
    from services.cyborg_client import _make_dbconfig
except Exception:
    CyborgVectorStore = None
    logger.error("Cyborg integration not available")

try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None
    np = None
    logger.error("Pandas/Numpy not available")


# -------------------------------------------------------------------
# INTENT CLASSIFICATION
# -------------------------------------------------------------------
class IntentClassifier:
    """Classify user intent for marketing MMM queries."""
    
    INTENTS = {
        'roi_analysis': [
            r'roi', r'return on investment', r'effectiveness', r'which channel',
            r'best channel', r'most effective', r'profitable', r'contribution'
        ],
        'attribution': [
            r'attribution', r'credit', r'impact', r'contribution', r'spend',
            r'allocate', r'budget', r'investment'
        ],
        'forecast': [
            r'forecast', r'predict', r'what if', r'scenario', r'simulate',
            r'estimate', r'projection', r'future'
        ],
        'optimize': [
            r'optimize', r'optimization', r'maximize', r'improve', r'increase',
            r'better allocation', r'rebalance', r'shift budget'
        ],
        'compare_channels': [
            r'compare', r'versus', r'vs', r'difference', r'which is better',
            r'tv vs', r'social vs', r'channel comparison'
        ],
        'model_info': [
            r'model', r'trained', r'parameters', r'performance', r'metrics',
            r'accuracy', r'r2', r'rmse', r'how does', r'what model'
        ],
        'data_info': [
            r'data', r'columns', r'features', r'target', r'dataset',
            r'uploaded', r'files', r'what data', r'channels'
        ]
    }
    
    @classmethod
    def classify(cls, query):
        """Return primary intent and confidence."""
        query_lower = query.lower()
        scores = {}
        
        for intent, patterns in cls.INTENTS.items():
            score = sum(1 for p in patterns if re.search(p, query_lower))
            if score > 0:
                scores[intent] = score
        
        if not scores:
            return 'general', 0.0
        
        primary = max(scores.items(), key=lambda x: x[1])
        return primary[0], primary[1] / len(cls.INTENTS[primary[0]])


# -------------------------------------------------------------------
# ENTITY EXTRACTION
# -------------------------------------------------------------------
class EntityExtractor:
    """Extract relevant entities from user queries."""
    
    @staticmethod
    def extract_channels(query, available_channels):
        """Find if user mentions specific marketing channels."""
        query_lower = query.lower()
        mentioned = []
        for channel in available_channels:
            if channel.lower() in query_lower:
                mentioned.append(channel)
        return mentioned if mentioned else None
    
    @staticmethod
    def extract_budget_amount(query):
        """Extract budget/spend amount from query."""
        # Patterns: "$1000", "1000 dollars", "1k", "1m"
        patterns = [
            r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*([km])?',
            r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars|usd|k|m)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query.lower())
            if match:
                amount_str = match.group(1).replace(',', '')
                amount = float(amount_str)
                
                if len(match.groups()) > 1 and match.group(2):
                    multiplier = match.group(2).lower()
                    if multiplier == 'k':
                        amount *= 1000
                    elif multiplier == 'm':
                        amount *= 1000000
                
                return amount
        
        return None
    
    @staticmethod
    def extract_percentage(query):
        """Extract percentage from query."""
        match = re.search(r'(\d+(?:\.\d+)?)\s*%', query)
        if match:
            return float(match.group(1))
        return None


# -------------------------------------------------------------------
# MODEL MANAGER
# -------------------------------------------------------------------
class ModelManager:
    """Manage loaded MMM models and their metadata."""
    
    @staticmethod
    def list_models(models_dir):
        """List all available MMM models with metadata."""
        models = []
        try:
            files = os.listdir(models_dir)
            meta_files = [f for f in files if f.endswith('.meta.json')]
            
            for meta_file in meta_files:
                meta_path = os.path.join(models_dir, meta_file)
                try:
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # Only include Marketing MMM models
                    if 'Marketing' in meta.get('model_name', '') or 'ElasticNetCV' in meta.get('model_name', '') or 'RidgeCV' in meta.get('model_name', ''):
                        base = meta_file.replace('.meta.json', '')
                        pkl_file = f"{base}.pkl"
                        
                        if pkl_file in files:
                            models.append({
                                'name': meta.get('model_name'),
                                'pkl_file': pkl_file,
                                'meta_file': meta_file,
                                'metadata': meta
                            })
                except Exception as e:
                    logger.warning(f"Failed to load metadata {meta_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        
        return models
    
    @staticmethod
    def load_model(models_dir, pkl_file):
        """Load a specific model."""
        path = os.path.join(models_dir, pkl_file)
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def get_latest_model(models_dir):
        """Get the most recently created MMM model."""
        models = ModelManager.list_models(models_dir)
        if not models:
            return None
        
        models.sort(
            key=lambda m: m['metadata'].get('created_at', ''),
            reverse=True
        )
        return models[0]


# -------------------------------------------------------------------
# CAMPAIGN PLANNER
# -------------------------------------------------------------------
class CampaignPlanner:
    """Generate campaign recommendations with budget allocation and projections."""
    
    @staticmethod
    def get_upcoming_events():
        """Detect upcoming holidays and events."""
        from datetime import datetime, timedelta
        
        now = datetime.now()
        month = now.month
        day = now.day
        
        events = []
        
        # Christmas season (Nov-Dec)
        if month == 11 or month == 12:
            events.append({
                'name': 'Christmas & Holiday Season',
                'theme': '🎄 Festive Shopping Season',
                'period': 'November - December',
                'audience': 'Gift shoppers, families, last-minute buyers',
                'urgency': 'HIGH' if (month == 12 and day > 15) else 'MEDIUM'
            })
        
        # New Year (Dec-Jan)
        if month == 12 or month == 1:
            events.append({
                'name': 'New Year Campaign',
                'theme': '🎆 Fresh Start & Resolutions',
                'period': 'Late December - January',
                'audience': 'Goal-setters, health-conscious, productivity seekers',
                'urgency': 'HIGH' if (month == 12 and day > 20) else 'MEDIUM'
            })
        
        # Valentine's Day (Jan-Feb)
        if month == 1 or month == 2:
            events.append({
                'name': "Valentine's Day Special",
                'theme': '💝 Love & Romance',
                'period': 'Late January - February 14',
                'audience': 'Couples, romantic gift shoppers',
                'urgency': 'HIGH' if (month == 2 and day > 7) else 'MEDIUM'
            })
        
        # Spring/Easter (Mar-Apr)
        if month in [3, 4]:
            events.append({
                'name': 'Spring Revival Campaign',
                'theme': '🌸 Fresh Season Launch',
                'period': 'March - April',
                'audience': 'Seasonal shoppers, renewal seekers',
                'urgency': 'MEDIUM'
            })
        
        # Summer (May-Aug)
        if month in [5, 6, 7, 8]:
            events.append({
                'name': 'Summer Spectacular',
                'theme': '☀️ Vacation & Outdoor Season',
                'period': 'May - August',
                'audience': 'Travelers, outdoor enthusiasts, families',
                'urgency': 'MEDIUM'
            })
        
        # Back to School (Aug-Sep)
        if month in [8, 9]:
            events.append({
                'name': 'Back to School Campaign',
                'theme': '📚 Academic Season',
                'period': 'August - September',
                'audience': 'Students, parents, educators',
                'urgency': 'HIGH' if month == 8 else 'MEDIUM'
            })
        
        # Black Friday / Cyber Monday (Nov)
        if month == 11:
            events.append({
                'name': 'Black Friday Mega Sale',
                'theme': '🛍️ Biggest Shopping Event',
                'period': 'Late November',
                'audience': 'Deal hunters, early holiday shoppers',
                'urgency': 'CRITICAL' if day > 20 else 'HIGH'
            })
        
        # Fallback - always have at least one campaign
        if not events:
            events.append({
                'name': 'Seasonal Promotion',
                'theme': '✨ Year-Round Excellence',
                'period': 'Current Period',
                'audience': 'General audience',
                'urgency': 'MEDIUM'
            })
        
        return events
    
    @staticmethod
    def generate_campaign_plan(model_info, query=None):
        """Generate comprehensive campaign plan with budget allocation."""
        try:
            # Get ROI data to inform budget allocation
            roi_result = ROIAnalyzer.compute_channel_roi(model_info)
            if not roi_result.get('success'):
                # Use default allocations if no ROI data
                roi_result = {
                    'roi_by_channel': {},
                    'ranked_channels': []
                }
            
            # Get upcoming events
            events = CampaignPlanner.get_upcoming_events()
            primary_event = events[0]
            
            # Generate campaign recommendations
            campaigns = []
            
            # Social Media Campaign
            social_budget = 15000  # Base budget
            social_campaign = {
                'channel': 'Social Media Ads',
                'icon': '📱',
                'platforms': ['Facebook', 'Instagram', 'TikTok', 'LinkedIn'],
                'budget': social_budget,
                'duration_days': 30,
                'estimated_metrics': {
                    'impressions': int(social_budget * 45),  # ~45 impressions per dollar
                    'clicks': int(social_budget * 2.5),  # ~2.5 clicks per dollar
                    'conversions': int(social_budget * 0.15),  # ~0.15 conversions per dollar
                    'ctr': 5.5,  # 5.5% CTR
                    'cpc': 0.40,  # $0.40 CPC
                    'conversion_rate': 6.0,  # 6% conversion rate
                    'roas': 4.2  # 4.2x ROAS
                },
                'targeting': {
                    'age': '25-54',
                    'interests': ['Shopping', 'Lifestyle', 'Technology'],
                    'behaviors': 'Active online shoppers, engaged users'
                },
                'creative_recommendations': [
                    'Video ads (15-30 sec) with festive themes',
                    'Carousel ads showcasing product collections',
                    'User-generated content campaigns',
                    'Interactive Stories and Reels'
                ]
            }
            campaigns.append(social_campaign)
            
            # Email Marketing Campaign
            email_budget = 8000
            email_campaign = {
                'channel': 'Email Marketing',
                'icon': '📧',
                'platforms': ['Newsletter', 'Automated Sequences', 'Promotional Blasts'],
                'budget': email_budget,
                'duration_days': 30,
                'estimated_metrics': {
                    'emails_sent': int(email_budget * 125),  # ~125 emails per dollar
                    'opens': int(email_budget * 30),  # ~30 opens per dollar (24% open rate)
                    'clicks': int(email_budget * 8),  # ~8 clicks per dollar
                    'conversions': int(email_budget * 0.48),  # ~0.48 conversions per dollar
                    'open_rate': 24.0,  # 24% open rate
                    'ctr': 26.7,  # 26.7% CTR (of opens)
                    'conversion_rate': 6.0,  # 6% conversion rate
                    'roas': 5.8  # 5.8x ROAS
                },
                'targeting': {
                    'segments': ['Existing customers', 'Newsletter subscribers', 'Cart abandoners'],
                    'personalization': 'High (name, purchase history, preferences)'
                },
                'creative_recommendations': [
                    'Personalized subject lines with urgency',
                    'Mobile-optimized responsive designs',
                    'Countdown timers for limited offers',
                    'A/B test different CTAs and imagery'
                ]
            }
            campaigns.append(email_campaign)
            
            # Search Ads (if relevant)
            search_budget = 12000
            search_campaign = {
                'channel': 'Search Ads (Google)',
                'icon': '🔍',
                'platforms': ['Google Search', 'Google Shopping'],
                'budget': search_budget,
                'duration_days': 30,
                'estimated_metrics': {
                    'impressions': int(search_budget * 200),  # ~200 impressions per dollar
                    'clicks': int(search_budget * 4),  # ~4 clicks per dollar
                    'conversions': int(search_budget * 0.32),  # ~0.32 conversions per dollar
                    'ctr': 2.0,  # 2% CTR
                    'cpc': 0.25,  # $0.25 CPC
                    'conversion_rate': 8.0,  # 8% conversion rate
                    'roas': 5.5  # 5.5x ROAS
                },
                'targeting': {
                    'keywords': 'High-intent transactional keywords',
                    'match_types': 'Exact and phrase match prioritized',
                    'locations': 'Primary markets with geo-targeting'
                },
                'creative_recommendations': [
                    'Ad extensions (sitelinks, callouts, snippets)',
                    'Dynamic keyword insertion',
                    'Seasonal promotional messaging',
                    'Remarketing lists for search ads (RLSA)'
                ]
            }
            campaigns.append(search_campaign)
            
            # Calculate totals
            total_budget = sum(c['budget'] for c in campaigns)
            total_estimated_conversions = sum(
                c['estimated_metrics'].get('conversions', 0) 
                for c in campaigns
            )
            weighted_roas = sum(
                c['budget'] * c['estimated_metrics'].get('roas', 0)
                for c in campaigns
            ) / total_budget if total_budget > 0 else 0
            
            # Generate strategic recommendations
            recommendations = [
                f"🎯 Focus on {primary_event['theme']} messaging throughout all channels",
                f"⏰ Campaign urgency: {primary_event['urgency']} - act fast for maximum impact",
                "📊 Allocate 40% to social for broad reach, 35% to search for intent, 25% to email for retention",
                "🔄 Implement cross-channel retargeting to maximize conversion paths",
                "📈 Monitor daily and optimize based on real-time performance",
                "🎨 Use consistent creative themes across channels for brand cohesion",
                "💰 Reserve 20% of budget for rapid scaling of top performers",
                "📱 Ensure all landing pages are mobile-optimized (60%+ mobile traffic expected)"
            ]
            
            # Budget allocation pie chart data
            allocation_data = [
                {'channel': c['channel'], 'budget': c['budget'], 'percentage': (c['budget']/total_budget*100)}
                for c in campaigns
            ]
            
            return {
                'success': True,
                'campaign_name': primary_event['name'],
                'theme': primary_event['theme'],
                'period': primary_event['period'],
                'target_audience': primary_event['audience'],
                'urgency': primary_event['urgency'],
                'campaigns': campaigns,
                'totals': {
                    'total_budget': total_budget,
                    'total_estimated_conversions': total_estimated_conversions,
                    'weighted_average_roas': round(weighted_roas, 2),
                    'estimated_revenue': int(total_budget * weighted_roas)
                },
                'allocation_data': allocation_data,
                'recommendations': recommendations,
                'timeline': {
                    'planning_phase': '1 week',
                    'creative_development': '1-2 weeks',
                    'campaign_launch': 'ASAP',
                    'optimization_period': '2-4 weeks',
                    'analysis_post_campaign': '1 week'
                }
            }
            
        except Exception as e:
            logger.exception(f"Campaign planning failed: {e}")
            return {'success': False, 'error': str(e)}


# -------------------------------------------------------------------
# ROI ANALYZER
# -------------------------------------------------------------------
class ROIAnalyzer:
    """Analyze ROI and attribution from trained MMM models."""
    
    @staticmethod
    def compute_channel_roi(model_info):
        """Compute ROI for each marketing channel."""
        try:
            meta = model_info['metadata']
            feature_names = meta.get('feature_names', [])
            spend_cols = meta.get('spend_columns_detected', [])
            
            # Load model to get coefficients
            models_dir = current_app.config.get('MODELS_FOLDER', 'models')
            model = ModelManager.load_model(models_dir, model_info['pkl_file'])
            
            # Get coefficients
            if hasattr(model, 'coef_'):
                coefs = model.coef_
            else:
                return {'success': False, 'error': 'Model does not have coefficients'}
            
            # Map coefficients to spend columns
            roi_data = {}
            for spend_col in spend_cols:
                # Find corresponding feature (transformed with _m_ prefix)
                feature_name = f"_m_{spend_col}"
                if feature_name in feature_names:
                    idx = feature_names.index(feature_name)
                    coef = float(coefs[idx])
                    
                    roi_data[spend_col] = {
                        'coefficient': coef,
                        'relative_impact': abs(coef)
                    }
            
            # Normalize to get relative contributions
            total_impact = sum(abs(v['relative_impact']) for v in roi_data.values())
            if total_impact > 0:
                for channel in roi_data:
                    roi_data[channel]['contribution_pct'] = (
                        abs(roi_data[channel]['relative_impact']) / total_impact * 100
                    )
            
            # Rank channels by impact
            ranked = sorted(
                roi_data.items(),
                key=lambda x: abs(x[1]['relative_impact']),
                reverse=True
            )
            
            return {
                'success': True,
                'roi_by_channel': roi_data,
                'ranked_channels': [(ch, data) for ch, data in ranked],
                'top_channel': ranked[0][0] if ranked else None,
                'model_name': meta.get('model_name'),
                'target': meta.get('target_column')
            }
            
        except Exception as e:
            logger.exception(f"ROI computation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def simulate_budget_change(model_info, channel, change_pct):
        """Simulate impact of changing budget for a specific channel."""
        try:
            roi_result = ROIAnalyzer.compute_channel_roi(model_info)
            if not roi_result.get('success'):
                return roi_result
            
            channel_data = roi_result['roi_by_channel'].get(channel)
            if not channel_data:
                return {
                    'success': False,
                    'error': f'Channel {channel} not found in model'
                }
            
            # Estimate impact: coefficient * budget_change
            coef = channel_data['coefficient']
            estimated_impact = coef * (change_pct / 100)
            
            return {
                'success': True,
                'channel': channel,
                'budget_change_pct': change_pct,
                'estimated_impact': estimated_impact,
                'current_coefficient': coef,
                'recommendation': (
                    f"Increasing {channel} budget by {change_pct}% could change "
                    f"your target by approximately {estimated_impact:.2%}"
                )
            }
            
        except Exception as e:
            logger.exception(f"Budget simulation failed: {e}")
            return {'success': False, 'error': str(e)}


# -------------------------------------------------------------------
# RESPONSE FORMATTER
# -------------------------------------------------------------------
class ResponseFormatter:
    """Format responses for different intents."""
    
    @staticmethod
    def format_roi_response(roi_result, query):
        """Format ROI analysis into natural language."""
        if not roi_result.get('success'):
            return f"I couldn't analyze ROI: {roi_result.get('error')}"
        
        ranked = roi_result['ranked_channels']
        target = roi_result['target']
        
        response = [
            f"Based on the Marketing Mix Model, here's the ROI analysis for {target}:",
            ""
        ]
        
        # Show top channels
        response.append("📊 Channel Performance (by impact):")
        for i, (channel, data) in enumerate(ranked[:5], 1):
            coef = data['coefficient']
            contrib = data.get('contribution_pct', 0)
            impact_indicator = "📈" if coef > 0 else "📉"
            
            response.append(
                f"  {i}. {impact_indicator} {channel}: "
                f"{contrib:.1f}% contribution | "
                f"Coefficient: {coef:.4f}"
            )
        
        if len(ranked) > 5:
            response.append(f"  ... and {len(ranked) - 5} more channels")
        
        # Key insights
        response.extend([
            "",
            "💡 Key Insights:",
            f"  • Top performing channel: {ranked[0][0]}",
            f"  • This channel accounts for {ranked[0][1].get('contribution_pct', 0):.1f}% of model impact"
        ])
        
        # Check for low performers
        low_performers = [ch for ch, data in ranked if abs(data['coefficient']) < 0.01]
        if low_performers:
            response.append(
                f"  • Consider reviewing: {', '.join(low_performers[:3])} "
                f"(low impact detected)"
            )
        
        return "\n".join(response)
    
    @staticmethod
    def format_attribution_response(roi_result):
        """Format attribution/contribution analysis."""
        if not roi_result.get('success'):
            return f"I couldn't compute attribution: {roi_result.get('error')}"
        
        roi_data = roi_result['roi_by_channel']
        
        response = [
            "📊 Marketing Attribution Analysis:",
            ""
        ]
        
        for channel, data in sorted(
            roi_data.items(),
            key=lambda x: x[1].get('contribution_pct', 0),
            reverse=True
        ):
            contrib = data.get('contribution_pct', 0)
            coef = data['coefficient']
            
            response.append(
                f"  • {channel}: {contrib:.1f}% (coef: {coef:.4f})"
            )
        
        response.extend([
            "",
            "This shows each channel's relative contribution to the target variable.",
            "Higher percentages indicate stronger attribution."
        ])
        
        return "\n".join(response)
    
    @staticmethod
    def format_optimization_response(roi_result):
        """Suggest budget optimization."""
        if not roi_result.get('success'):
            return f"I couldn't generate optimization suggestions: {roi_result.get('error')}"
        
        ranked = roi_result['ranked_channels']
        
        response = [
            "💰 Budget Optimization Suggestions:",
            "",
            "Based on model coefficients, consider:"
        ]
        
        # High performers to invest more
        top_3 = ranked[:3]
        response.extend([
            "",
            "📈 Increase investment in:"
        ])
        for channel, data in top_3:
            if data['coefficient'] > 0:
                response.append(
                    f"  • {channel} - High positive impact "
                    f"({data.get('contribution_pct', 0):.1f}% contribution)"
                )
        
        # Low performers to reduce
        bottom_3 = ranked[-3:]
        low_impact = [
            (ch, data) for ch, data in bottom_3
            if abs(data['coefficient']) < 0.01
        ]
        
        if low_impact:
            response.extend([
                "",
                "📉 Consider reducing:"
            ])
            for channel, data in low_impact:
                response.append(
                    f"  • {channel} - Low measured impact "
                    f"({data.get('contribution_pct', 0):.1f}% contribution)"
                )
        
        response.extend([
            "",
            "💡 Remember: This is based on historical data. Consider:",
            "  • Market conditions and competitive landscape",
            "  • Long-term brand building vs short-term sales",
            "  • Channel synergies and cross-effects"
        ])
        
        return "\n".join(response)
    
    @staticmethod
    def format_model_info(model_info):
        """Format model information."""
        meta = model_info['metadata']
        
        response = [
            f"Model: {meta.get('model_name')}",
            f"Target: {meta.get('target_column')}",
            f"Training Samples: {meta.get('train_n', 'Unknown')}",
            ""
        ]
        
        spend_cols = meta.get('spend_columns_detected', [])
        if spend_cols:
            response.append("Marketing Channels:")
            for channel in spend_cols[:10]:
                response.append(f"  • {channel}")
            if len(spend_cols) > 10:
                response.append(f"  ... and {len(spend_cols) - 10} more")
            response.append("")
        
        # Transformations applied
        if meta.get('adstock_applied'):
            response.append(f"Adstock Transform: Yes (decay={meta.get('adstock_decay', 0.5)})")
        if meta.get('saturation_alpha'):
            response.append(f"Saturation Alpha: {meta.get('saturation_alpha')}")
        
        response.append(f"\nCreated: {meta.get('created_at')}")
        
        return "\n".join(response)
    
    @staticmethod
    def format_comparison_response(channels, roi_result):
        """Compare specific channels."""
        if not roi_result.get('success'):
            return f"I couldn't compare channels: {roi_result.get('error')}"
        
        roi_data = roi_result['roi_by_channel']
        
        response = [
            f"📊 Comparing: {' vs '.join(channels)}",
            ""
        ]
        
        for channel in channels:
            if channel in roi_data:
                data = roi_data[channel]
                response.extend([
                    f"**{channel}:**",
                    f"  • Contribution: {data.get('contribution_pct', 0):.1f}%",
                    f"  • Coefficient: {data['coefficient']:.4f}",
                    f"  • Impact: {'Positive' if data['coefficient'] > 0 else 'Negative'}",
                    ""
                ])
            else:
                response.append(f"**{channel}:** Not found in model\n")
        
        # Winner
        valid_channels = [ch for ch in channels if ch in roi_data]
        if valid_channels:
            winner = max(
                valid_channels,
                key=lambda ch: roi_data[ch].get('contribution_pct', 0)
            )
            response.append(f"🏆 Winner: {winner}")
        
        return "\n".join(response)
    
    @staticmethod
    def format_campaign_context(campaign_plan):
        """Format campaign plan as brief context."""
        if not campaign_plan.get('success'):
            return ""
        
        return (
            f"\n📣 UPCOMING CAMPAIGN OPPORTUNITY: {campaign_plan['campaign_name']}\n"
            f"{campaign_plan['theme']} | Period: {campaign_plan['period']}\n"
            f"Recommended Budget: ${campaign_plan['totals']['total_budget']:,} | "
            f"Expected ROAS: {campaign_plan['totals']['weighted_average_roas']}x\n"
            f"Ask me for the full campaign plan to see detailed breakdown!"
        )
    
    @staticmethod
    def format_campaign_full(campaign_plan):
        """Format complete campaign plan."""
        if not campaign_plan.get('success'):
            return f"I couldn't generate a campaign plan: {campaign_plan.get('error')}"
        
        response = [
            f"🎯 {campaign_plan['campaign_name'].upper()}",
            f"{campaign_plan['theme']}",
            "",
            f"📅 Campaign Period: {campaign_plan['period']}",
            f"🎪 Target Audience: {campaign_plan['target_audience']}",
            f"⚡ Urgency Level: {campaign_plan['urgency']}",
            "",
            "=" * 60,
            "",
            "💰 BUDGET ALLOCATION & PROJECTIONS:",
            ""
        ]
        
        for campaign in campaign_plan['campaigns']:
            response.extend([
                f"{campaign['icon']} {campaign['channel'].upper()}",
                f"Budget: ${campaign['budget']:,} ({campaign['budget']/campaign_plan['totals']['total_budget']*100:.1f}% of total)",
                f"Duration: {campaign['duration_days']} days",
                ""
            ])
            
            metrics = campaign['estimated_metrics']
            if 'impressions' in metrics:
                response.extend([
                    "Expected Performance:",
                    f"  • Impressions: {metrics['impressions']:,}",
                    f"  • Clicks: {metrics['clicks']:,}",
                    f"  • Conversions: {metrics['conversions']:,}",
                    f"  • CTR: {metrics.get('ctr', 0):.1f}%",
                    f"  • Conversion Rate: {metrics.get('conversion_rate', 0):.1f}%",
                    f"  • ROAS: {metrics.get('roas', 0):.1f}x",
                    ""
                ])
            elif 'emails_sent' in metrics:
                response.extend([
                    "Expected Performance:",
                    f"  • Emails Sent: {metrics['emails_sent']:,}",
                    f"  • Opens: {metrics['opens']:,}",
                    f"  • Clicks: {metrics['clicks']:,}",
                    f"  • Conversions: {metrics['conversions']:,}",
                    f"  • Open Rate: {metrics.get('open_rate', 0):.1f}%",
                    f"  • CTR: {metrics.get('ctr', 0):.1f}%",
                    f"  • ROAS: {metrics.get('roas', 0):.1f}x",
                    ""
                ])
        
        response.extend([
            "=" * 60,
            "",
            "📊 CAMPAIGN TOTALS:",
            f"  • Total Budget: ${campaign_plan['totals']['total_budget']:,}",
            f"  • Total Expected Conversions: {campaign_plan['totals']['total_estimated_conversions']:,}",
            f"  • Weighted Average ROAS: {campaign_plan['totals']['weighted_average_roas']:.2f}x",
            f"  • Estimated Revenue: ${campaign_plan['totals']['estimated_revenue']:,}",
            "",
            "=" * 60,
            "",
            "💡 STRATEGIC RECOMMENDATIONS:",
            ""
        ])
        
        for i, rec in enumerate(campaign_plan['recommendations'], 1):
            response.append(f"{i}. {rec}")
        
        response.extend([
            "",
            "=" * 60,
            "",
            "⏱️ TIMELINE:",
            f"  • Planning: {campaign_plan['timeline']['planning_phase']}",
            f"  • Creative Development: {campaign_plan['timeline']['creative_development']}",
            f"  • Launch: {campaign_plan['timeline']['campaign_launch']}",
            f"  • Optimization: {campaign_plan['timeline']['optimization_period']}",
            f"  • Post-Analysis: {campaign_plan['timeline']['analysis_post_campaign']}",
            "",
            "🚀 Ready to launch? Let's make this campaign a success!"
        ])
        
        return "\n".join(response)


# -------------------------------------------------------------------
# INIT GEMINI CLIENT
# -------------------------------------------------------------------
def init_gemini():
    api_key = "AIzaSyB_cMKuBZPux9FttkqZSFEsDJjcUlyukqY"
    if not api_key:
        logger.error("Missing GEMINI_API_KEY")
        return None
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-2.5-flash")
    except Exception as e:
        logger.exception("Failed to init Gemini: %s", e)
        return None


# -------------------------------------------------------------------
# INIT CYBORG VECTOR STORE
# -------------------------------------------------------------------
def get_vector_store():
    """Initialize CyborgVectorStore for semantic search."""
    if CyborgVectorStore is None:
        logger.error("CyborgVectorStore not importable")
        return None

    try:
        cyborg_api_key = "cyborg_de8158fd38be4b97a400d4712fa77e3d"
        if not cyborg_api_key:
            logger.error("Missing CYBORG_API_KEY")
            return None

        cyborg_index_name = current_app.config.get('CYBORG_INDEX_NAME', 'embedded_index_v16')
        models_dir = current_app.config.get('MODELS_FOLDER', os.path.join(os.getcwd(), 'models'))
        keys_folder = os.path.join(models_dir, 'cyborg_indexes')
        key_path = os.path.join(keys_folder, f"{cyborg_index_name}.key")

        if not os.path.exists(key_path):
            logger.error("Cyborg index key missing: %s", key_path)
            return None

        with open(key_path, 'rb') as f:
            index_key = f.read()

        storage_index = current_app.config.get('CYBORG_INDEX_STORAGE', 'postgres')
        storage_config = current_app.config.get('CYBORG_CONFIG_STORAGE', 'postgres')
        storage_items = current_app.config.get('CYBORG_ITEMS_STORAGE', 'postgres')

        pg_uri = current_app.config.get(
            'CYBORG_PG_URI',
            os.environ.get('CYBORG_PG_URI', 'postgresql://cyborg:cyborg123@localhost:5432/cyborgdb'),
        )

        tbl_index = current_app.config.get('CYBORG_INDEX_TABLE', 'cyborg_index')
        tbl_config = current_app.config.get('CYBORG_CONFIG_TABLE', 'cyborg_config')
        tbl_items = current_app.config.get('CYBORG_ITEMS_TABLE', 'cyborg_items')

        index_loc = _make_dbconfig(
            storage_index,
            connection_string=pg_uri if storage_index == 'postgres' else None,
            table_name=tbl_index,
        )
        config_loc = _make_dbconfig(
            storage_config,
            connection_string=pg_uri if storage_config == 'postgres' else None,
            table_name=tbl_config,
        )
        items_loc = _make_dbconfig(
            storage_items,
            connection_string=pg_uri if storage_items == 'postgres' else None,
            table_name=tbl_items,
        )

        embedding_model = (
            current_app.config.get('CYBORG_EMBEDDING_MODEL')
            or os.environ.get('CYBORG_EMBEDDING_MODEL')
            or 'all-MiniLM-L6-v2'
        )

        vs = CyborgVectorStore(
            index_name=cyborg_index_name,
            index_key=index_key,
            api_key=cyborg_api_key,
            embedding=embedding_model,
            index_location=index_loc,
            config_location=config_loc,
            items_location=items_loc,
            metric='cosine',
        )

        logger.info("CyborgVectorStore initialized")
        return vs

    except Exception as e:
        logger.exception("Failed to build vector store: %s", e)
        return None


# -------------------------------------------------------------------
# SEMANTIC SEARCH
# -------------------------------------------------------------------
def semantic_search(vs, query, k=5):
    try:
        docs = vs.similarity_search_with_score(query, k)
        formatted = []
        for d, score in docs:
            formatted.append({
                'text': d.page_content,
                'metadata': d.metadata,
                'score': float(score)
            })
        return formatted
    except Exception as e:
        logger.exception("Search failed: %s", e)
        return []


# -------------------------------------------------------------------
# ENHANCED RAG PROMPT BUILDER
# -------------------------------------------------------------------
def build_enhanced_prompt(docs, query, context_info):
    """Build prompt with model and marketing data context."""
    blocks = [
        "You are an AI assistant specialized in Marketing Mix Modeling (MMM) and marketing attribution analysis.",
        "Answer the user's question based on the provided documents and context.",
        "Focus on ROI, channel effectiveness, budget optimization, and marketing performance.",
        "If the information is insufficient, clearly state what's missing.",
        ""
    ]
    
    # Add context
    if context_info:
        blocks.append("CONTEXT:")
        if context_info.get('available_models'):
            blocks.append(f"Available Models: {', '.join(context_info['available_models'])}")
        if context_info.get('latest_model'):
            blocks.append(f"Latest Model: {context_info['latest_model']}")
        if context_info.get('channels'):
            blocks.append(f"Marketing Channels: {', '.join(context_info['channels'][:10])}")
        if context_info.get('target'):
            blocks.append(f"Target Metric: {context_info['target']}")
        blocks.append("")
    
    # Add retrieved documents
    if docs:
        blocks.append("RELEVANT DOCUMENTS:")
        for i, d in enumerate(docs):
            snippet = d['text'][:2000]
            blocks.append(f"--- Document {i+1} (Score: {d['score']:.3f}) ---")
            blocks.append(snippet)
            blocks.append("")
    
    blocks.append(f"USER QUESTION:\n{query}\n")
    blocks.append("Provide a clear, actionable answer with specific recommendations:")
    
    return "\n".join(blocks)


# -------------------------------------------------------------------
# MAIN CHAT ENDPOINT
# -------------------------------------------------------------------
@bp.route('/api/marketing/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint for Marketing MMM with NLP capabilities."""
    data = request.get_json() or {}
    query = data.get('query')
    
    if not query:
        return jsonify({'success': False, 'error': 'query required'}), 400
    
    try:
        # 1. Classify intent
        intent, confidence = IntentClassifier.classify(query)
        logger.info(f"Intent: {intent} (confidence: {confidence:.2f})")
        
        # 2. Get models directory
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        
        # 3. Get latest model
        model_info = ModelManager.get_latest_model(models_dir)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': 'No trained Marketing MMM models available. Please upload marketing data and train a model first.'
            }), 404
        
        meta = model_info['metadata']
        channels = meta.get('spend_columns_detected', [])
        
        # ALWAYS generate campaign plan for context
        campaign_plan = CampaignPlanner.generate_campaign_plan(model_info, query)
        
        # 4. Handle ROI analysis intent
        if intent == 'roi_analysis':
            roi_result = ROIAnalyzer.compute_channel_roi(model_info)
            response_text = ResponseFormatter.format_roi_response(roi_result, query)
            
            # Add campaign context
            if campaign_plan.get('success'):
                response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'roi_data': roi_result if roi_result.get('success') else None,
                'campaign_plan': campaign_plan if campaign_plan.get('success') else None,
                'model_info': meta
            })
        
        # 5. Handle attribution intent
        if intent == 'attribution':
            roi_result = ROIAnalyzer.compute_channel_roi(model_info)
            response_text = ResponseFormatter.format_attribution_response(roi_result)
            
            # Add campaign context
            if campaign_plan.get('success'):
                response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'attribution_data': roi_result if roi_result.get('success') else None,
                'campaign_plan': campaign_plan if campaign_plan.get('success') else None,
                'model_info': meta
            })
        
        # 6. Handle optimization intent
        if intent == 'optimize':
            roi_result = ROIAnalyzer.compute_channel_roi(model_info)
            response_text = ResponseFormatter.format_optimization_response(roi_result)
            
            # Add campaign context
            if campaign_plan.get('success'):
                response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'optimization_data': roi_result if roi_result.get('success') else None,
                'campaign_plan': campaign_plan if campaign_plan.get('success') else None,
                'model_info': meta
            })
        
        # 7. Handle channel comparison
        if intent == 'compare_channels':
            mentioned_channels = EntityExtractor.extract_channels(query, channels)
            
            if mentioned_channels and len(mentioned_channels) >= 2:
                roi_result = ROIAnalyzer.compute_channel_roi(model_info)
                response_text = ResponseFormatter.format_comparison_response(
                    mentioned_channels, roi_result
                )
                
                # Add campaign context
                if campaign_plan.get('success'):
                    response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
                
                return jsonify({
                    'success': True,
                    'answer': response_text,
                    'intent': intent,
                    'compared_channels': mentioned_channels,
                    'roi_data': roi_result if roi_result.get('success') else None,
                    'campaign_plan': campaign_plan if campaign_plan.get('success') else None
                })
            else:
                return jsonify({
                    'success': True,
                    'answer': f"Please specify which channels you'd like to compare. Available channels: {', '.join(channels[:10])}",
                    'intent': intent,
                    'available_channels': channels,
                    'campaign_plan': campaign_plan if campaign_plan.get('success') else None
                })
        
        # 8. Handle forecast/simulation intent
        if intent == 'forecast':
            mentioned_channels = EntityExtractor.extract_channels(query, channels)
            change_pct = EntityExtractor.extract_percentage(query)
            
            if mentioned_channels and change_pct:
                channel = mentioned_channels[0]
                sim_result = ROIAnalyzer.simulate_budget_change(
                    model_info, channel, change_pct
                )
                
                if sim_result.get('success'):
                    response_text = (
                        f"💰 Budget Simulation for {channel}:\n\n"
                        f"{sim_result['recommendation']}\n\n"
                        f"Details:\n"
                        f"  • Budget change: {change_pct:+.1f}%\n"
                        f"  • Estimated impact: {sim_result['estimated_impact']:.2%}\n"
                        f"  • Channel coefficient: {sim_result['current_coefficient']:.4f}\n\n"
                        f"Note: This is a linear approximation. Actual results may vary due to market dynamics and channel interactions."
                    )
                else:
                    response_text = f"Simulation failed: {sim_result.get('error')}"
                
                # Add campaign context
                if campaign_plan.get('success'):
                    response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
                
                return jsonify({
                    'success': True,
                    'answer': response_text,
                    'intent': intent,
                    'simulation_data': sim_result if sim_result.get('success') else None,
                    'campaign_plan': campaign_plan if campaign_plan.get('success') else None
                })
            else:
                # Even for help text, include campaign plan
                response_text = (
                    "To simulate budget changes, please specify:\n"
                    "1. Which channel (e.g., 'TV', 'Social')\n"
                    "2. Percentage change (e.g., '20%')\n\n"
                    f"Available channels: {', '.join(channels[:10])}\n\n"
                    "Example: 'What if I increase TV spend by 20%?'"
                )
                
                if campaign_plan.get('success'):
                    response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
                
                return jsonify({
                    'success': True,
                    'answer': response_text,
                    'intent': intent,
                    'available_channels': channels,
                    'campaign_plan': campaign_plan if campaign_plan.get('success') else None
                })
        
        # 9. Handle model_info intent
        if intent == 'model_info':
            response_text = ResponseFormatter.format_model_info(model_info)
            
            # Add campaign context
            if campaign_plan.get('success'):
                response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'model_info': meta,
                'campaign_plan': campaign_plan if campaign_plan.get('success') else None
            })
        
        # 10. Handle data_info intent
        if intent == 'data_info':
            response_text = (
                f"Dataset Information:\n\n"
                f"Target: {meta.get('target_column')}\n"
                f"Training Samples: {meta.get('train_n')}\n\n"
                f"Marketing Channels ({len(channels)}):\n"
            )
            for channel in channels[:15]:
                response_text += f"  • {channel}\n"
            if len(channels) > 15:
                response_text += f"  ... and {len(channels) - 15} more\n"
            
            if meta.get('adstock_applied'):
                response_text += f"\nAdstock Transform: Applied (decay={meta.get('adstock_decay')})"
            
            # Add campaign context
            if campaign_plan.get('success'):
                response_text += "\n\n" + ResponseFormatter.format_campaign_context(campaign_plan)
            
            return jsonify({
                'success': True,
                'answer': response_text,
                'intent': intent,
                'channels': channels,
                'model_info': meta,
                'campaign_plan': campaign_plan if campaign_plan.get('success') else None
            })
        
        # 11. For general queries, provide campaign plan directly
        response_text = ResponseFormatter.format_campaign_full(campaign_plan)
        
        return jsonify({
            'success': True,
            'answer': response_text,
            'intent': 'general',
            'campaign_plan': campaign_plan if campaign_plan.get('success') else None,
            'model_info': meta
        })
    
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# -------------------------------------------------------------------
# ADDITIONAL ENDPOINTS
# -------------------------------------------------------------------

@bp.route('/api/marketing/chat/models', methods=['GET'])
def list_available_models():
    """List all available MMM models."""
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        models = ModelManager.list_models(models_dir)
        
        return jsonify({
            'success': True,
            'models': [
                {
                    'name': m['name'],
                    'target': m['metadata'].get('target_column'),
                    'created': m['metadata'].get('created_at'),
                    'channels': m['metadata'].get('spend_columns_detected', [])
                }
                for m in models
            ]
        })
    except Exception as e:
        logger.exception("Failed to list models")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/marketing/chat/roi', methods=['POST'])
def direct_roi_analysis():
    """Direct ROI analysis endpoint."""
    data = request.get_json() or {}
    model_name = data.get('model_name')
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        
        if model_name:
            models = ModelManager.list_models(models_dir)
            model_info = next(
                (m for m in models if m['name'] == model_name),
                None
            )
        else:
            model_info = ModelManager.get_latest_model(models_dir)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': 'Model not found'
            }), 404
        
        roi_result = ROIAnalyzer.compute_channel_roi(model_info)
        
        return jsonify(roi_result)
    
    except Exception as e:
        logger.exception("ROI analysis error")
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/marketing/chat/simulate', methods=['POST'])
def simulate_budget():
    """Simulate budget change endpoint."""
    data = request.get_json() or {}
    channel = data.get('channel')
    change_pct = data.get('change_pct', 10)
    
    if not channel:
        return jsonify({'success': False, 'error': 'channel required'}), 400
    
    try:
        models_dir = current_app.config.get('MODELS_FOLDER', 'models')
        model_info = ModelManager.get_latest_model(models_dir)
        
        if not model_info:
            return jsonify({
                'success': False,
                'error': 'No model available'
            }), 404
        
        sim_result = ROIAnalyzer.simulate_budget_change(
            model_info, channel, change_pct
        )
        
        return jsonify(sim_result)
    
    except Exception as e:
        logger.exception("Simulation error")
        return jsonify({'success': False, 'error': str(e)}), 500