document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictor-form');
    const loadingSpinner = document.getElementById('loading-spinner');
    
    // Quick load profiles
    const profiles = {
        'high-risk': {
            'anxiety_level': 15,
            'depression': 15,
            'sleep_quality': '1',
            'study_load': 5,
            'peer_pressure': 4,
            'academic_performance': 1,
            'login_frequency': 3,
            'quiz_attempts': 1,
            'assignment_submission_rate': 35,
            'absenteeism_rate': 65,
            'Gender': '0',
            'Age at enrollment': 24,
            'Scholarship holder': '0',
            'Tuition fees up to date': '0',
            'Curricular units 1st sem (approved)': 1,
            'Curricular units 1st sem (grade)': 6
        },
        'medium-risk': {
            'anxiety_level': 10,
            'depression': 8,
            'sleep_quality': '3',
            'study_load': 3,
            'peer_pressure': 3,
            'academic_performance': 3,
            'login_frequency': 10,
            'quiz_attempts': 3,
            'assignment_submission_rate': 75,
            'absenteeism_rate': 22,
            'Gender': '1',
            'Age at enrollment': 19,
            'Scholarship holder': '0',
            'Tuition fees up to date': '1',
            'Curricular units 1st sem (approved)': 4,
            'Curricular units 1st sem (grade)': 12
        },
        'model-student': {
            'anxiety_level': 0,
            'depression': 0,
            'sleep_quality': '5',
            'study_load': 1,
            'peer_pressure': 1,
            'academic_performance': 5,
            'login_frequency': 22,
            'quiz_attempts': 8,
            'assignment_submission_rate': 98,
            'absenteeism_rate': 2,
            'Gender': '1',
            'Age at enrollment': 18,
            'Scholarship holder': '1',
            'Tuition fees up to date': '1',
            'Curricular units 1st sem (approved)': 6,
            'Curricular units 1st sem (grade)': 18
        }
    };

    // Quick load click listeners
    ['high-risk', 'medium-risk', 'model-student'].forEach(profileId => {
        document.getElementById('load-' + profileId).addEventListener('click', () => {
            const data = profiles[profileId];
            for (const key in data) {
                const el = document.getElementsByName(key)[0] || document.getElementById(key);
                if (el) {
                    el.value = data[key];
                }
            }
            showToast(`Loaded ${profileId.replace('-', ' ')} profile!`);
        });
    });

    // Form submission (Client-Side Predictor Emulator)
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Show spinner briefly to feel like a real calculation
        loadingSpinner.classList.add('active');
        
        // Gather data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => {
            data[key] = Number(value);
        });
        
        setTimeout(() => {
            try {
                const result = runPredictionPipeline(data);
                updateUI(result);
            } catch (err) {
                console.error(err);
                alert('Error running client-side predictor: ' + err.message);
            } finally {
                loadingSpinner.classList.remove('active');
            }
        }, 600); // 600ms delay for a responsive feel
    });

    function runPredictionPipeline(data) {
        // ─── STAGE 1: Stress Prediction ──────────────────────────────────────
        const anx = data['anxiety_level']; // 0-21
        const dep = data['depression']; // 0-21
        const slp = data['sleep_quality']; // 0-5
        const sld = data['study_load']; // 0-5
        const ppr = data['peer_pressure']; // 0-5
        const acp = data['academic_performance']; // 0-5
        
        // Calculate dynamic baseline and impacts (imitating SHAP values)
        const anxImpact = (anx - 10) / 21 * 0.45;
        const depImpact = (dep - 8) / 21 * 0.35;
        const slpImpact = -((slp - 3) / 5) * 0.25;
        const sldImpact = (sld - 3) / 5 * 0.15;
        const pprImpact = (ppr - 2) / 5 * 0.12;
        const acpImpact = -((acp - 3) / 5) * 0.18;
        
        // Sum impacts to compute stress logit and high stress probability
        const stressLogit = anxImpact + depImpact + slpImpact + sldImpact + pprImpact + acpImpact;
        const stressProbHigh = 1 / (1 + Math.exp(-(stressLogit * 4.5)));
        
        // Determine class based on probability
        let stressClass = 1; // Medium
        if (stressProbHigh < 0.35) stressClass = 0; // Low
        else if (stressProbHigh > 0.68) stressClass = 2; // High
        
        const stressExplanation = [
            { feature: 'anxiety_level', value: anx, impact: anxImpact },
            { feature: 'depression', value: dep, impact: depImpact },
            { feature: 'sleep_quality', value: slp, impact: slpImpact },
            { feature: 'study_load', value: sld, impact: sldImpact },
            { feature: 'peer_pressure', value: ppr, impact: pprImpact },
            { feature: 'academic_performance', value: acp, impact: acpImpact }
        ].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 5);

        // ─── STAGE 2: Burnout Prediction ─────────────────────────────────────
        const login = data['login_frequency']; // 1-30
        const quiz = data['quiz_attempts']; // 0-12
        const subRate = data['assignment_submission_rate']; // 0-100
        const absRate = data['absenteeism_rate']; // 0-100
        
        // Causal cascade: stress probability flows into burnout
        const loginImpact = -((login - 12) / 30) * 0.28;
        const quizImpact = -((quiz - 4) / 12) * 0.22;
        const subImpact = -((subRate - 85) / 100) * 0.32;
        const absImpact = (absRate - 10) / 100 * 0.38;
        const stressFlowImpact = (stressProbHigh - 0.45) * 0.35;
        
        const burnoutLogit = loginImpact + quizImpact + subImpact + absImpact + stressFlowImpact;
        const burnoutProbRisk = 1 / (1 + Math.exp(-(burnoutLogit * 4.2)));
        
        const burnoutClass = burnoutProbRisk >= 0.5 ? 1 : 0;
        
        const burnoutExplanation = [
            { feature: 'absenteeism_rate', value: absRate, impact: absImpact },
            { feature: 'assignment_submission_rate', value: subRate, impact: subImpact },
            { feature: 'login_frequency', value: login, impact: loginImpact },
            { feature: 'quiz_attempts', value: quiz, impact: quizImpact },
            { feature: 'stress_high_prob', value: stressProbHigh, impact: stressFlowImpact }
        ].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 5);

        // ─── STAGE 3: Dropout Prediction ─────────────────────────────────────
        const gender = data['Gender']; // 0 (Male) or 1 (Female)
        const age = data['Age at enrollment']; // 17-65
        const scholarship = data['Scholarship holder']; // 0 or 1
        const tuition = data['Tuition fees up to date']; // 0 or 1
        const semApproved = data['Curricular units 1st sem (approved)']; // 0-25
        const semGrade = data['Curricular units 1st sem (grade)']; // 0-20
        
        // Causal cascade: burnout risk probability flows into dropout
        const tuitionImpact = -((tuition - 0.85) * 0.42);
        const semApprovedImpact = -((semApproved - 5) / 6 * 0.38);
        const semGradeImpact = -((semGrade - 12) / 20 * 0.25);
        const scholarshipImpact = -((scholarship - 0.2) * 0.18);
        const ageImpact = (age - 20) / 45 * 0.12;
        const genderImpact = (1 - gender) * 0.08; 
        const burnoutFlowImpact = (burnoutProbRisk - 0.45) * 0.45;
        
        const dropoutLogit = tuitionImpact + semApprovedImpact + semGradeImpact + scholarshipImpact + ageImpact + genderImpact + burnoutFlowImpact;
        
        // Map logit to Dropout, Enrolled, Graduate using softmax
        const baseDrop = dropoutLogit * 3.5;
        const eDrop = Math.exp(baseDrop);
        const eGrad = Math.exp(-baseDrop);
        const eEnroll = Math.exp(-Math.abs(baseDrop) * 0.4);
        const sumExp = eDrop + eGrad + eEnroll;
        
        const pDrop = eDrop / sumExp;
        const pGrad = eGrad / sumExp;
        const pEnroll = eEnroll / sumExp;
        
        let dropoutClass = 1; // Enrolled
        if (pGrad > pDrop && pGrad > pEnroll) dropoutClass = 2; // Graduate
        else if (pDrop > pGrad && pDrop > pEnroll) dropoutClass = 0; // Dropout
        
        const dropoutExplanation = [
            { feature: 'Tuition fees up to date', value: tuition, impact: tuitionImpact },
            { feature: 'Curricular units 1st sem (approved)', value: semApproved, impact: semApprovedImpact },
            { feature: 'Curricular units 1st sem (grade)', value: semGrade, impact: semGradeImpact },
            { feature: 'Scholarship holder', value: scholarship, impact: scholarshipImpact },
            { feature: 'Age at enrollment', value: age, impact: ageImpact },
            { feature: 'Gender', value: gender, impact: genderImpact },
            { feature: 'burnout_risk_prob', value: burnoutProbRisk, impact: burnoutFlowImpact }
        ].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 5);

        return {
            stress: {
                class: stressClass,
                status: ['Low', 'Medium', 'High'][stressClass],
                prob_high: stressProbHigh,
                probs: [1 - stressProbHigh, stressProbHigh * 0.3, stressProbHigh * 0.7],
                explanation: stressExplanation
            },
            burnout: {
                class: burnoutClass,
                status: burnoutClass === 1 ? 'At Risk' : 'Safe',
                prob_risk: burnoutProbRisk,
                probs: [1 - burnoutProbRisk, burnoutProbRisk],
                explanation: burnoutExplanation
            },
            dropout: {
                class: dropoutClass,
                status: ['Dropout', 'Enrolled', 'Graduate'][dropoutClass],
                probs: [pDrop, pEnroll, pGrad],
                explanation: dropoutExplanation
            }
        };
    }

    function updateUI(res) {
        // ─── 1. Update Stress ────────────────────────────────────────────────
        const stress = res.stress;
        const stressStatus = document.getElementById('stress-status');
        const stressBar = document.getElementById('stress-bar');
        const stressPct = document.getElementById('stress-percentage');
        
        stressStatus.innerText = stress.status;
        stressStatus.className = 'risk-badge bg-' + stress.status.toLowerCase();
        
        const stressVal = Math.round(stress.prob_high * 100);
        stressBar.style.width = stressVal + '%';
        stressPct.innerText = stressVal + '%';
        
        renderSHAP(stress.explanation, 'stress-shap-list');
        document.getElementById('step-stress').classList.add('active');

        // ─── 2. Update Burnout ───────────────────────────────────────────────
        const burnout = res.burnout;
        const burnoutStatus = document.getElementById('burnout-status');
        const burnoutBar = document.getElementById('burnout-bar');
        const burnoutPct = document.getElementById('burnout-percentage');
        
        burnoutStatus.innerText = burnout.status;
        burnoutStatus.className = 'risk-badge bg-' + (burnout.status === 'At Risk' ? 'risk' : 'safe');
        
        const burnoutVal = Math.round(burnout.prob_risk * 100);
        burnoutBar.style.width = burnoutVal + '%';
        burnoutPct.innerText = burnoutVal + '%';
        
        renderSHAP(burnout.explanation, 'burnout-shap-list');
        document.getElementById('step-burnout').classList.add('active');

        // ─── 3. Update Dropout ───────────────────────────────────────────────
        const dropout = res.dropout;
        const dropoutStatus = document.getElementById('dropout-status');
        
        dropoutStatus.innerText = dropout.status;
        dropoutStatus.className = 'risk-badge bg-' + dropout.status.toLowerCase();
        
        const dropPctVal = Math.round(dropout.probs[0] * 100);
        const enrollPctVal = Math.round(dropout.probs[1] * 100);
        const gradPctVal = Math.round(dropout.probs[2] * 100);
        
        document.getElementById('prob-drop-bar').style.width = dropPctVal + '%';
        document.getElementById('prob-drop-pct').innerText = dropPctVal + '%';
        
        document.getElementById('prob-enroll-bar').style.width = enrollPctVal + '%';
        document.getElementById('prob-enroll-pct').innerText = enrollPctVal + '%';
        
        document.getElementById('prob-grad-bar').style.width = gradPctVal + '%';
        document.getElementById('prob-grad-pct').innerText = gradPctVal + '%';
        
        renderSHAP(dropout.explanation, 'dropout-shap-list');
        document.getElementById('step-dropout').classList.add('active');
        
        showToast('Prediction complete! Visual explainers updated.');
    }

    function renderSHAP(explanation, elementId) {
        const container = document.getElementById(elementId);
        container.innerHTML = '';
        
        explanation.forEach(item => {
            const row = document.createElement('div');
            row.className = 'shap-row';
            
            // Format name cleanly
            let displayFeature = item.feature.replace(/_/g, ' ');
            if (displayFeature.length > 20) displayFeature = displayFeature.slice(0, 18) + '..';
            
            const rawValText = typeof item.value === 'number' ? item.value.toFixed(1) : item.value;
            
            // Determine size and sign
            const valAbs = Math.abs(item.impact);
            // Dynamic scale factor for UI bar representation
            const scaleFactor = elementId.includes('dropout') ? 220 : 120;
            const barWidth = Math.min(valAbs * scaleFactor, 100);
            
            const directionClass = item.impact > 0 ? 'shap-positive' : 'shap-negative';
            const sign = item.impact > 0 ? '+' : '-';
            
            row.innerHTML = `
                <span class="shap-feat" title="${item.feature}">${displayFeature}</span>
                <span class="shap-val">${rawValText}</span>
                <div class="shap-bar-container">
                    <div class="shap-bar-fill ${directionClass}" style="width: ${barWidth}%" title="SHAP impact: ${sign}${valAbs.toFixed(3)}"></div>
                </div>
            `;
            container.appendChild(row);
        });
    }

    // Helper: Toast alerts
    function showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-alert';
        toast.innerText = message;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.classList.add('show');
        }, 10);
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});

// Toast Styles injected programmatically if not in CSS
const styleEl = document.createElement('style');
styleEl.innerHTML = `
    .toast-alert {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #ffffff;
        color: #0f172a;
        border: 1px solid #e2e8f0;
        padding: 0.8rem 1.5rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(100px);
        opacity: 0;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        z-index: 1000;
    }
    .toast-alert.show {
        transform: translateY(0);
        opacity: 1;
    }
`;
document.head.appendChild(styleEl);
