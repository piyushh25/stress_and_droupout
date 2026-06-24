import os
import numpy as np
import pandas as pd

def load_dataset_1():
    """Loads Dataset 1: Student Stress Factors (from archive3)"""
    path = os.path.join('data', 'archive3', 'StressLevelDataset.csv')
    if not os.path.exists(path):
        # Fallback to archive5 if archive3 not found
        path = os.path.join('data', 'archive5', 'StressLevelDataset.csv')
    return pd.read_csv(path)

def load_dataset_2():
    """Loads Dataset 2: Stress and Well-being Data (from archive5)"""
    path = os.path.join('data', 'archive5', 'Stress_Dataset.csv')
    if not os.path.exists(path):
        # Fallback to archive4 if archive5 not found
        path = os.path.join('data', 'archive4', 'Stress Dataset.csv')
    return pd.read_csv(path)

def generate_lms_dataset(num_records=2400):
    """Generates synthetic LMS engagement dataset (Dataset 3) matching paper description"""
    np.random.seed(42)
    
    # Generate engagement metrics
    login_frequency = np.random.normal(loc=12, scale=4, size=num_records)  # Logins per week
    login_frequency = np.clip(login_frequency, 1, 30).astype(int)
    
    quiz_attempts = np.random.poisson(lam=4, size=num_records)
    quiz_attempts = np.clip(quiz_attempts, 0, 12)
    
    assignment_submission_rate = np.random.normal(loc=82, scale=12, size=num_records)  # Percentage
    assignment_submission_rate = np.clip(assignment_submission_rate, 10, 100)
    
    time_on_platform = login_frequency * np.random.normal(loc=25, scale=8, size=num_records)  # Minutes per week
    time_on_platform = np.clip(time_on_platform, 10, 800).astype(int)
    
    forum_participation = np.random.poisson(lam=1.5, size=num_records)
    
    video_completion_rate = np.random.uniform(low=10, high=100, size=num_records)  # Percentage
    
    grade_trend = np.random.choice([-1, 0, 1], p=[0.25, 0.45, 0.3], size=num_records)
    
    absenteeism_rate = np.random.normal(loc=15, scale=10, size=num_records)  # Percentage of classes missed
    absenteeism_rate = np.clip(absenteeism_rate, 0, 80)
    
    help_seeking_behavior = np.random.poisson(lam=0.8, size=num_records)
    help_seeking_behavior = np.clip(help_seeking_behavior, 0, 5)
    
    # Calculate burnout risk based on realistic rules (causal links)
    # Lower logins, lower submissions, higher absenteeism -> high risk
    risk_score = (
        (30 - login_frequency) * 1.5 + 
        (100 - assignment_submission_rate) * 0.8 + 
        absenteeism_rate * 1.2 - 
        forum_participation * 3 - 
        help_seeking_behavior * 4 + 
        (grade_trend == -1).astype(int) * 15
    )
    
    # Convert risk score to probability, then binary label
    prob = 1 / (1 + np.exp(-(risk_score - 40) / 15))
    burnout_risk = (np.random.rand(num_records) < prob).astype(int)
    
    df_lms = pd.DataFrame({
        'login_frequency': login_frequency,
        'quiz_attempts': quiz_attempts,
        'assignment_submission_rate': assignment_submission_rate,
        'time_on_platform': time_on_platform,
        'forum_participation': forum_participation,
        'video_completion_rate': video_completion_rate,
        'grade_trend': grade_trend,
        'absenteeism_rate': absenteeism_rate,
        'help_seeking_behavior': help_seeking_behavior,
        'burnout_risk': burnout_risk
    })
    
    # Save the synthetic dataset
    os.makedirs('data', exist_ok=True)
    df_lms.to_csv(os.path.join('data', 'student_lms_activity.csv'), index=False)
    print(f"Generated synthetic LMS dataset at data/student_lms_activity.csv | Rows: {len(df_lms)}")
    return df_lms

def load_dataset_3():
    """Loads or generates Dataset 3: Student Stress Monitoring (LMS Logs)"""
    path = os.path.join('data', 'student_lms_activity.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    else:
        return generate_lms_dataset()

def load_dataset_4():
    """Loads Dataset 4: Predict Students' Dropout and Academic Success (from archive6)"""
    path = os.path.join('data', 'archive6', 'dataset.csv')
    return pd.read_csv(path)

if __name__ == '__main__':
    # Test loading
    d1 = load_dataset_1()
    d2 = load_dataset_2()
    d3 = load_dataset_3()
    d4 = load_dataset_4()
    print("Testing data loading:")
    print(f"  Dataset 1: {d1.shape}")
    print(f"  Dataset 2: {d2.shape}")
    print(f"  Dataset 3: {d3.shape}")
    print(f"  Dataset 4: {d4.shape}")
