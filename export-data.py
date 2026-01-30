#!/usr/bin/env python3
"""
Export all study data from the SQLite database for analysis.

This script exports:
- All participants data with demographics (Prolific users only)
- All annotations (ratings) from Prolific users
- Dialogue assignment counts
- Gold standard performance
- Attention check results
- Merged datasets for analysis

Output formats: CSV and JSON
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime
from pathlib import Path

# Database path
DB_PATH = 'data/study.db'
EXPORT_DIR = 'data/exports'
PROLIFIC_ONLY = True  # Set to False to include all users


def create_export_directory():
    """Create timestamped export directory."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    export_path = Path(EXPORT_DIR) / timestamp
    export_path.mkdir(parents=True, exist_ok=True)
    return export_path


def export_participants(conn, export_path):
    """Export participants table with all demographics and study metadata."""
    query = """
    SELECT
        id as participant_id,
        prolific_id,
        study_id,
        session_id,
        is_prolific_user,
        assigned_dialogues,
        assigned_gold_standards,
        order_dialogues,
        attention_check_assignments,
        attention_check_results,
        attention_check_failures,
        completed_count,
        created_at,
        age,
        native_english,
        movie_frequency,
        chatbot_frequency,
        demographics_completed
    FROM participants
    """
    
    if PROLIFIC_ONLY:
        query += "WHERE is_prolific_user = 1 "
    
    query += "ORDER BY created_at"

    df = pd.read_sql_query(query, conn)

    # Parse JSON columns
    json_columns = ['assigned_dialogues', 'assigned_gold_standards', 'order_dialogues',
                    'attention_check_assignments', 'attention_check_results']

    for col in json_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: json.loads(x) if pd.notna(x) and x else None)

    # Add exclusion flag based on attention check failures
    # Note: Using default max_failures_allowed = 1 from config
    df['should_exclude'] = df['attention_check_failures'] > 1

    # Export to CSV and JSON
    df.to_csv(export_path / 'participants.csv', index=False)
    df.to_json(export_path / 'participants.json', orient='records', indent=2)

    print(f"✅ Exported {len(df)} participants" + (" (Prolific only)" if PROLIFIC_ONLY else ""))
    return df


def export_annotations(conn, export_path):
    """Export all annotations (ratings) data."""
    query = """
    SELECT
        id as annotation_id,
        participant_id,
        dialogue_id,
        is_gold_standard,
        is_prolific_user,
        accuracy,
        novelty,
        interaction_adequacy,
        explainability,
        cui_adaptability,
        cui_understanding,
        cui_response_quality,
        cui_attentiveness,
        perceived_ease_of_use,
        perceived_usefulness,
        user_control,
        transparency,
        cui_humanness,
        cui_rapport,
        trust_confidence,
        satisfaction,
        intention_to_use,
        intention_to_purchase,
        time_spent,
        timestamp
    FROM annotations
    """
    
    if PROLIFIC_ONLY:
        query += "WHERE is_prolific_user = 1 "
    
    query += "ORDER BY timestamp"

    df = pd.read_sql_query(query, conn)

    # Export to CSV and JSON
    df.to_csv(export_path / 'annotations.csv', index=False)
    df.to_json(export_path / 'annotations.json', orient='records', indent=2)

    print(f"✅ Exported {len(df)} annotations" + (" (Prolific only)" if PROLIFIC_ONLY else ""))
    return df


def export_dialogue_counts(conn, export_path):
    """Export dialogue annotation counts."""
    query = """
    SELECT
        dialogue_id,
        annotation_count
    FROM dialogue_counts
    ORDER BY annotation_count DESC, dialogue_id
    """

    df = pd.read_sql_query(query, conn)

    # Export to CSV and JSON
    df.to_csv(export_path / 'dialogue_counts.csv', index=False)
    df.to_json(export_path / 'dialogue_counts.json', orient='records', indent=2)

    print(f"✅ Exported {len(df)} dialogue counts")
    return df


def export_merged_data(annotations_df, participants_df, export_path):
    """Create merged dataset with annotations + participant demographics."""
    # Select relevant participant columns for merging
    participant_cols = ['participant_id', 'prolific_id', 'is_prolific_user',
                       'age', 'native_english', 'movie_frequency', 'chatbot_frequency',
                       'attention_check_failures', 'should_exclude']

    participants_subset = participants_df[participant_cols].copy()

    # Merge annotations with participant data
    merged_df = annotations_df.merge(
        participants_subset,
        on='participant_id',
        how='left',
        suffixes=('', '_participant')
    )

    # Drop duplicate is_prolific_user column if it exists
    if 'is_prolific_user_participant' in merged_df.columns:
        merged_df = merged_df.drop(columns=['is_prolific_user_participant'])

    # Export full merged dataset
    merged_df.to_csv(export_path / 'annotations_with_demographics.csv', index=False)
    merged_df.to_json(export_path / 'annotations_with_demographics.json', orient='records', indent=2)

    print(f"✅ Exported merged dataset: {len(merged_df)} rows")

    # No additional filtered datasets needed
    return merged_df


def export_gold_standard_performance(annotations_df, participants_df, export_path):
    """Export gold standard annotations for quality analysis."""
    gold_df = annotations_df[annotations_df['is_gold_standard'] == True].copy()

    # Merge with participant data
    participant_cols = ['participant_id', 'prolific_id', 'is_prolific_user',
                       'attention_check_failures', 'should_exclude']
    participants_subset = participants_df[participant_cols].copy()

    gold_merged = gold_df.merge(
        participants_subset,
        on='participant_id',
        how='left',
        suffixes=('', '_participant')
    )

    # Drop duplicate columns
    if 'is_prolific_user_participant' in gold_merged.columns:
        gold_merged = gold_merged.drop(columns=['is_prolific_user_participant'])

    gold_merged.to_csv(export_path / 'gold_standard_annotations.csv', index=False)
    gold_merged.to_json(export_path / 'gold_standard_annotations.json', orient='records', indent=2)

    print(f"✅ Exported {len(gold_merged)} gold standard annotations" + (" (Prolific only)" if PROLIFIC_ONLY else ""))
    return gold_merged


def export_attention_check_details(participants_df, export_path):
    """Export detailed attention check results."""
    attention_data = []

    for _, participant in participants_df.iterrows():
        if participant['attention_check_results']:
            results = participant['attention_check_results']

            for dialogue_id, check_result in results.items():
                attention_data.append({
                    'participant_id': participant['participant_id'],
                    'prolific_id': participant['prolific_id'],
                    'is_prolific_user': participant['is_prolific_user'],
                    'dialogue_id': dialogue_id,
                    'check_id': check_result.get('check_id'),
                    'expected_answer': check_result.get('expected_answer'),
                    'user_answer': check_result.get('user_answer'),
                    'passed': check_result.get('passed'),
                    'total_failures': participant['attention_check_failures'],
                    'should_exclude': participant['should_exclude']
                })

    if attention_data:
        attention_df = pd.DataFrame(attention_data)
        attention_df.to_csv(export_path / 'attention_check_results.csv', index=False)
        attention_df.to_json(export_path / 'attention_check_results.json', orient='records', indent=2)
        print(f"✅ Exported {len(attention_df)} attention check results")
        return attention_df
    else:
        print("⚠️  No attention check results found")
        return pd.DataFrame()


def generate_summary_statistics(participants_df, annotations_df, export_path):
    """Generate summary statistics about the study."""
    summary = {
        'export_timestamp': datetime.now().isoformat(),
        'database_path': DB_PATH,
        'prolific_only': PROLIFIC_ONLY,
        'participants': {
            'total': len(participants_df),
            'prolific_users': len(participants_df[participants_df['is_prolific_user'] == True]),
            'test_users': len(participants_df[participants_df['is_prolific_user'] == False]),
            'completed_demographics': len(participants_df[participants_df['demographics_completed'] == 1]),
            'with_attention_failures': len(participants_df[participants_df['attention_check_failures'] > 0]),
            'excluded_by_attention_checks': len(participants_df[participants_df['should_exclude'] == True]),
        },
        'annotations': {
            'total': len(annotations_df),
            'regular_dialogues': len(annotations_df[annotations_df['is_gold_standard'] == False]),
            'gold_standards': len(annotations_df[annotations_df['is_gold_standard'] == True]),
            'from_prolific_users': len(annotations_df[annotations_df['is_prolific_user'] == True]),
            'from_test_users': len(annotations_df[annotations_df['is_prolific_user'] == False]),
        },
        'time_metrics': {
            'avg_time_per_annotation': float(annotations_df['time_spent'].mean()) if 'time_spent' in annotations_df.columns else None,
            'median_time_per_annotation': float(annotations_df['time_spent'].median()) if 'time_spent' in annotations_df.columns else None,
            'total_time_hours': float(annotations_df['time_spent'].sum() / 3600) if 'time_spent' in annotations_df.columns else None,
        },
        'data_quality': {
            'participants_passed_attention_checks': len(participants_df[participants_df['should_exclude'] == False]),
            'participants_failed_attention_checks': len(participants_df[participants_df['should_exclude'] == True]),
            'high_quality_annotations': len(annotations_df[
                (annotations_df['is_prolific_user'] == True) &
                (annotations_df.merge(participants_df[['participant_id', 'should_exclude']], on='participant_id')['should_exclude'] == False)
            ]),
        },
        'rating_statistics': {}
    }

    # Calculate mean and std for each rating dimension
    rating_columns = [
        'accuracy', 'novelty', 'interaction_adequacy', 'explainability',
        'cui_adaptability', 'cui_understanding', 'cui_response_quality', 'cui_attentiveness',
        'perceived_ease_of_use', 'perceived_usefulness', 'user_control', 'transparency',
        'cui_humanness', 'cui_rapport', 'trust_confidence', 'satisfaction',
        'intention_to_use', 'intention_to_purchase'
    ]

    for col in rating_columns:
        if col in annotations_df.columns:
            summary['rating_statistics'][col] = {
                'mean': float(annotations_df[col].mean()),
                'std': float(annotations_df[col].std()),
                'min': int(annotations_df[col].min()),
                'max': int(annotations_df[col].max()),
                'median': float(annotations_df[col].median()),
            }

    # Save summary
    with open(export_path / 'summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✅ Generated summary statistics")
    return summary


def print_export_summary(export_path, summary):
    """Print a nice summary of the export."""
    print("\n" + "="*60)
    print("📊 EXPORT SUMMARY" + (" (PROLIFIC USERS ONLY)" if PROLIFIC_ONLY else ""))
    print("="*60)
    print(f"Export location: {export_path}")
    print(f"\nParticipants:")
    print(f"  • Total: {summary['participants']['total']}")
    print(f"  • Prolific users: {summary['participants']['prolific_users']}")
    print(f"  • Test users: {summary['participants']['test_users']}")
    print(f"  • Excluded (attention checks): {summary['participants']['excluded_by_attention_checks']}")
    print(f"\nAnnotations:")
    print(f"  • Total: {summary['annotations']['total']}")
    print(f"  • Regular dialogues: {summary['annotations']['regular_dialogues']}")
    print(f"  • Gold standards: {summary['annotations']['gold_standards']}")
    print(f"  • High quality (Prolific + passed checks): {summary['data_quality']['high_quality_annotations']}")
    print(f"\nFiles exported:")
    exported_files = list(export_path.glob('*'))
    for file in sorted(exported_files):
        print(f"  • {file.name}")
    print("="*60)


def main():
    """Main export function."""
    print("🚀 Starting data export...\n")

    # Check if database exists
    if not os.path.exists(DB_PATH):
        print(f"❌ Error: Database not found at {DB_PATH}")
        print("   Please run this script from the backend/scripts/ directory")
        return

    # Create export directory
    export_path = create_export_directory()
    print(f"📁 Export directory: {export_path}\n")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)

    try:
        # Export all tables
        participants_df = export_participants(conn, export_path)
        annotations_df = export_annotations(conn, export_path)
        export_dialogue_counts(conn, export_path)

        # Export merged datasets
        print()
        export_merged_data(annotations_df, participants_df, export_path)

        # Export gold standard performance
        print()
        export_gold_standard_performance(annotations_df, participants_df, export_path)

        # Export attention check details
        print()
        export_attention_check_details(participants_df, export_path)

        # Generate summary statistics
        print()
        summary = generate_summary_statistics(participants_df, annotations_df, export_path)

        # Print final summary
        print_export_summary(export_path, summary)

        print(f"\n✅ Export completed successfully!")
        print(f"   All data saved to: {export_path}")

    except Exception as e:
        print(f"\n❌ Error during export: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


if __name__ == "__main__":
    main()
