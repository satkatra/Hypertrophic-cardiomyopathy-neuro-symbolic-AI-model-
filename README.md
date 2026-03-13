# Hypertrophic-cardiomyopathy-neuro-symbolic-AI-model-
7 iteration of a model to build a neurosymbolic HCM model also contains a 1d CNN model if needed

Table of Contents
Section 1  Project Ideation and Initial Research
Section 2  Problem Definition and Engineering Goal
Section 3  Background Research and Literature Review
Section 4  Dataset Discovery and Acquisition
Section 5  Development Environment Setup
Section 6  Label Extraction Strategy
Section 7  Model 1 — Random Forest Baseline
Section 8  Model 2 — 1D Convolutional Neural Network
Section 9  Model 3 — Neuro-Symbolic V1 (Recording Split)
Section 10  Model 4 — Neuro-Symbolic V2 (Patient Split)
Section 11  Model 5 — Neuro-Symbolic V3 (Focal Loss)
Section 12  Model 6 — Neuro-Symbolic V4 (Residual CNN)
Section 13  Model 7 — Neuro-Symbolic V5 (Attention + Final)
Section 14  Visualization Generation
Section 15  Results Analysis and Interpretation
Section 16  Limitations and Future Work
Section 17  Project Summary and Conclusions

Section 1 — Project Ideation and Initial Research
ENTRY 1.1
DATE: 10/15/2025 to 10/25/2025


1.1  How I Got the Idea
I kept reading about young athletes dying suddenly during games and practices. It kept coming up in sports news and I wanted to understand why. After some digging I found that the most common cause is a condition called hypertrophic cardiomyopathy, or HCM, where the heart walls grow too thick. What got me was that most of these kids had no idea anything was wrong. HCM affects roughly 1 in 500 people and the majority are never diagnosed. Also I found out my grandpa has HCM himself and I dedicated myself to trying to figure out to prevent people from finding out too late.

The thing that really stuck with me was learning that HCM actually does leave a trace on an ECG — the electrical signals look different. Cardiologists can see it, but the changes are subtle and get missed a lot, especially in early stages. So the question I kept coming back to was: could AI do better? Could a model trained on thousands of ECGs learn to spot these patterns more reliably than a human reader doing a quick scan?

1.2  Brainstorming Different Approaches
Before committing to anything, I spent a few days thinking through different ways to build this. Here is what I considered:

Approach A — Pure rules: Just code up the known clinical ECG thresholds (Sokolow-Lyon voltage, Cornell voltage, etc.) and flag anyone who exceeds them. I rejected this pretty quickly. It is too rigid. Borderline cases get missed, and it can't learn anything the original cardiologists didn't already know.
Approach B — Pure deep learning CNN: Train a neural network on the raw ECG signal and let it figure out the patterns itself. This seemed promising, but the problem is you end up with a black box. There is no way to explain to a doctor why it flagged someone, and that matters a lot in medicine. I kept this as part of the design but not the whole thing.
Approach C — Neuro-symbolic hybrid: Combine the CNN with the clinical rules I would have used in Approach A. The CNN learns the subtle stuff, the rules add interpretability. This is what I went with. It felt like the most honest design for a medical tool.
Approach D — Fine-tune a pre-trained ECG model: There are large ECG models already out there. But fine-tuning them requires serious compute and the proxy labels I was using would probably add noise rather than signal. I set this aside.

NOTE
The thing that kept coming back to me during brainstorming: a doctor is never going to trust a model that just says '78% HCM' with no explanation. If I wanted this to actually be useful, interpretability wasn't a nice-to-have — it was a requirement. That's what pushed me toward the neuro-symbolic design.


1.3  Why ECG and Not Something Else?
The obvious gold standard for HCM is an echocardiogram — you can literally see how thick the walls are. But it needs specialized equipment, a trained technician, and it costs a lot. ECGs are everywhere. They take 10 seconds, cost almost nothing, and you can get one at a school athletic physical, a rural clinic, or a sports camp. If AI can make ECG screening for HCM meaningfully more sensitive, the number of people that could reach is huge. That was the practical argument for focusing on ECG.

1.4  Questions I Was Trying to Answer
Can a model actually pick up on HCM-related ECG patterns that human readers miss?
Does combining learned CNN features with explicit clinical rules improve things over a pure deep learning approach?
What classification threshold makes sense clinically — and is there a good way to find it automatically?
Will the model generalize to patients it has never seen at all, not just recordings it hasn't seen?

1.5  Why This is Engineering, Not Science
When I looked at the ISEF categories I had to decide: is this science or engineering? The honest answer is engineering. I am not testing a hypothesis about cardiac physiology. I am designing and building a system, making specific choices about architecture and methodology, measuring whether those choices improve performance, and iterating. The output is a working prototype with quantified metrics. That is an engineering project. I documented it that way throughout.

Section 2 — Problem Definition and Engineering Goal
ENTRY 2.1
DATE: 11/10/2025 to 11/27/2025


2.1  What HCM Actually Is
HCM stands for hypertrophic cardiomyopathy. The heart muscle — specifically the left ventricular wall — grows thicker than it should. This thickening can block blood flow out of the heart, cause dangerous arrhythmias, and in the worst cases, lead to sudden cardiac death. It is genetic, it can show up at any age, and it is the leading cause of sudden cardiac death in people under 35, especially athletes.

What makes it so dangerous is how quietly it sits. Most people with HCM feel completely fine. They play sports, pass physicals, and have no idea there is a problem until something goes wrong. By that point it is often too late. There is no cure, but if it is caught early there are real options — medication, lifestyle changes, implanted defibrillators, and in some cases surgery. Early detection is not just helpful, it is life-saving.

2.2  What HCM Looks Like on an ECG
HCM changes the electrical activity of the heart in measurable ways. Here are the patterns I focused on:
Left ventricular hypertrophy (LVH) — the thicker muscle mass creates higher voltages in the precordial leads
Sokolow-Lyon criterion — S-wave depth in V1 plus R-wave height in V5 greater than 3.5 mV
Cornell voltage criterion — S in V1 plus R in aVL greater than 2.8 mV
T-wave inversion in lateral leads (V5, V6) — the repolarization pattern looks abnormal
Deep Q-waves — pseudoinfarction pattern sometimes seen in inferior or lateral leads
Left axis deviation

The tricky part is that none of these are unique to HCM. Athletes with enlarged hearts from training can show similar voltage patterns. Early disease may show none of these at all. This is exactly why a model that can learn subtle combinations of features has an advantage over any single rule.

2.3  Engineering Goal
GOAL STATEMENT
Design, build, and iteratively improve an AI prototype that can detect early signs of HCM
from 12-lead ECG data. Each model version must make a specific, measurable improvement
over the previous one. Final model must achieve AUC > 0.85 on a patient-level test set
using only publicly available, de-identified ECG data, running on standard CPU hardware.


2.4  Design Requirements I Set for Myself
Use only publicly available, de-identified data — no IRB needed, no privacy issues
Hit AUC > 0.85 on a patient-level test set (not recording-level — more on that later)
Run on a regular laptop CPU, no GPU — the point is accessibility
Keep it interpretable — the model's decisions need to connect to something a doctor recognizes
Use proper methodology: patient splits, validation monitoring, early stopping, fixed random seeds
Build at least 5 versions with documented, measurable improvements each time

Section 3 — Background Research and Literature Review
ENTRY 3.1
DATE: 12/15/2025 to 01/12/2026


3.1  Papers I Read Before Starting
I tried to read everything I could find before writing a single line of code. These are the papers that mattered most and what I actually took away from each.

[1] Ko, W.Y. et al. (2020). Detection of HCM Using a CNN-Enabled ECG Algorithm. Circulation: Arrhythmia and Electrophysiology.
This one was the most relevant find. Ko and colleagues trained a CNN on clinical ECG data and showed it could identify HCM better than manual reading — catching repolarization and voltage patterns that trained cardiologists missed. The finding that really stood out to me was that AI is not just faster than a human reader, it sees things humans genuinely cannot. That validated the whole premise of this project.
What it confirmed: CNNs applied to 12-lead ECGs can detect HCM-related patterns.
What it left open: the model is a black box. No clinical rule integration, no interpretability. A cardiologist cannot audit why it flagged a patient.
What I did differently: added the symbolic layer specifically to address this. My model gives a reason tied to known clinical criteria.

[2] Wagner, P. et al. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data.
This is the dataset I used. PTB-XL has 21,837 12-lead ECG recordings from 18,885 patients, collected in Germany between 1989 and 1996. Each recording comes with diagnostic annotations using SCP-ECG codes. It is the largest openly available ECG dataset in the world, fully de-identified, and available through PhysioNet and Kaggle. Reading through this paper also gave me the metadata structure I needed to understand which codes corresponded to hypertrophy-related conditions.
100 Hz recordings used — 1000 samples per 10-second recording across 12 leads
Relevant SCP codes: LVH, RVH, LVOLT, SEHYP — these became my proxy labels

[3] Strodthoff, N. et al. (2021). Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL. IEEE JBHI.
Super useful for methodology. This paper benchmarked multiple deep learning approaches on PTB-XL and established preprocessing standards. I followed their z-score normalization approach and their recommendation for patient-level splitting. This paper is also where I first understood concretely why recording-level splitting is a problem — they discuss it explicitly.

[4] He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
The ResNet paper. Read this before building Model 6 (NS V4). The core idea is that very deep networks stop learning because gradients vanish during backpropagation before reaching early layers. Residual connections — shortcuts that skip layers and add the input directly to the output — fix this. The shortcut gives gradients a free path back through the network. I adapted this for 1D ECG data in the V4 model.

[5] Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV.
Introduced focal loss for object detection where foreground objects are rare compared to background. Exactly the same imbalance problem I had: 88% healthy, 12% HCM. Standard BCE loss lets the model get away with always predicting healthy. Focal loss down-weights easy correct predictions and forces the model to pay attention to the hard cases. Read this before building Model 5 (NS V3).

[6] Xu, W. et al. (2024). An overview of treatments for hypertrophic cardiomyopathy. Frontiers in Cardiovascular Medicine.
This was more clinical background than technical. The key takeaway for my project: early diagnosis makes a real difference. Patients caught before a cardiac event have real treatment options. That kept reminding me why the precision-recall tradeoff actually matters — missing someone is not just a bad metric, it has consequences.

3.2  Gaps I Noticed in the Literature
After reading through everything, a few things stood out as missing:
Most published ECG AI models do not integrate clinical rules. They achieve high performance but offer no interpretability path for clinicians.
A lot of early-stage and student projects on ECG datasets use recording-level splits and report inflated metrics. Patient-level methodology is standard in published clinical AI research but rarely applied rigorously at the student level.
Almost nobody reports threshold optimization. The default 0.5 cutoff is arbitrary and clinically unjustified.
These gaps became the specific things I tried to address in this project.

Section 4 — Dataset Discovery and Acquisition
ENTRY 4.1
DATE: 1/25/2026 to 2/15/2026


4.1  Why I Chose PTB-XL
I looked at a few ECG datasets before settling on PTB-XL. My requirements were straightforward: large enough to actually train a deep learning model, fully de-identified so I could use it without any IRB concerns, publicly available, 12-lead format, and some form of diagnostic annotation that would let me extract HCM-related labels. PTB-XL hit every one of those. It is also the benchmark dataset in ECG deep learning research right now, so there was good published methodology to reference.
I downloaded it from Kaggle as a ZIP file (~1.8 GB). PhysioNet is the original host but Kaggle mirrors it and I found the download faster. After extraction I checked the folder structure against what the Wagner et al. paper described to make sure everything was there.

4.2  File Structure After Download
# After extracting the ZIP, this is what the directory looks like:
datasets/ptb-xl/
  ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/
    ptbxl_database.csv     # One row per recording — metadata + diagnostic codes
    scp_statements.csv     # Code lookup table: what each SCP code means
    records100/            # 100 Hz ECG recordings (.hea header + .dat data pairs)
      00000/
        00001_lr.hea
        00001_lr.dat
        ...
    records500/            # 500 Hz versions — I didn't use these (RAM constraints)

4.3  Verification Script — test_ecg.py
First thing I did was write a quick script to confirm the data loaded correctly. Before building anything complex, I wanted to see one ECG on screen and confirm the dimensions were what I expected.
# test_ecg.py
import wfdb
import matplotlib.pyplot as plt
 
BASE = 'datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
 
record = wfdb.rdrecord(f'{BASE}/records100/00000/00001_lr')
print('Shape:', record.p_signal.shape)  # should be (1000, 12)
 
# Plot lead I just to see the waveform looks right
plt.figure(figsize=(12, 3))
plt.plot(record.p_signal[:, 0])
plt.title('ECG Lead I - Record 00001')
plt.xlabel('Time (samples at 100Hz)')
plt.ylabel('Amplitude (mV)')
plt.tight_layout()
plt.show()

RESULT
Shape: (1000, 12) confirmed. That is 1000 time samples x 12 leads = 10 seconds at 100 Hz. The waveform plot looked like a normal ECG. Dataset loaded correctly.


4.4  Dataset at a Glance
Property
Value
Total Recordings
21,837
Total Unique Patients
18,885
Recording Duration
10 seconds
Sampling Rate Used
100 Hz (1000 samples per recording)
Leads
12-lead standard
HCM-Proxy Labels (after extraction)
2,449 (~11.2%)
Control Labels
19,388 (~88.8%)
Class Imbalance Ratio
~7.9 to 1


Section 5 — Development Environment Setup
ENTRY 5.1
DATE: 2/16/2026


5.1  Setting Up the Environment
I set up a Python virtual environment to keep project dependencies isolated. This is just good practice — you don't want package version conflicts bleeding across projects.
# Working directory for everything:
cd C:\Users\ashgo\GDrive-satkatra\HCM_AI_Project
 
# Create virtual environment
python -m venv hcm_env
 
# Activate it
# In PowerShell:
hcm_env\Scripts\activate
 
# In CMD (if PowerShell gives execution policy errors):
hcm_env\Scripts\activate.bat
 
# Fix PowerShell if scripts are blocked:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser

5.2  Libraries Installed
pip install wfdb pandas numpy scikit-learn torch torchvision matplotlib
 
# What each one does:
# wfdb        — reads PhysioNet's WFDB format (.hea + .dat file pairs)
# pandas      — data manipulation, CSV handling, label management
# numpy       — numerical operations on the ECG signal arrays
# scikit-learn — Random Forest, train/test splits, AUC, F1, etc.
# torch       — PyTorch, the deep learning framework for all CNN models
# matplotlib  — plotting ECG waveforms and results charts

5.3  Project File Structure
HCM_AI_Project/
  datasets/
    ptb-xl/                          # The full PTB-XL dataset
    hcm_labels.csv                   # My extracted HCM proxy labels
  hcm_env/                           # Python virtual environment
  test_ecg.py                        # Verify dataset loads correctly
  label_ecg.py                       # Extract HCM labels from SCP codes
  train_baseline.py                  # Model 1: Random Forest
  train_cnn.py                       # Model 2: 1D-CNN
  train_neurosymbolic_v1_recordingsplit.py
  train_neurosymbolic_v2_patientsplit.py
  train_neurosymbolic_v3_focal.py
  train_neurosymbolic_v4_residual.py
  train_neurosymbolic_v5_attention.py   # Final model
  generate_visualizations.py
  hcm_neurosymbolic_v5.pt               # Saved final model weights
  hcm_results_poster.png
  hcm_accuracy_comparison.png

5.4  Issues I Hit During Setup
PowerShell blocked script execution
Running hcm_env\Scripts\activate in PowerShell returned an error about execution policy. Fixed by running Set-ExecutionPolicy RemoteSigned -Scope CurrentUser, or just switching to CMD for that step.

Scripts exiting silently after pasting into Notepad
I pasted some Python scripts into Notepad on Windows and they would run but produce no output, or fail syntax checks silently. Turned out Notepad was mangling certain Unicode characters in the code. Solution: use VS Code for writing and editing scripts. If in doubt, verify syntax before running:
python -c "import py_compile; py_compile.compile('script.py', doraise=True)"

wfdb.dl_database crashed with 404 error
I tried using wfdb's built-in download function initially. It failed partway through with malformed URLs at the records100/records500 boundary — a known bug in wfdb. Kaggle download was clean and faster anyway.

Section 6 — Label Extraction Strategy
ENTRY 6.1
DATE: 2/17/2026 


6.1  The Label Problem
PTB-XL does not have an HCM label. That was the first real challenge. To train a supervised classifier, I needed to know which ECGs come from patients with HCM-like heart conditions. The dataset uses SCP-ECG codes, a European diagnostic coding system. Explicit HCM codes essentially don't appear in the dataset in meaningful numbers.

The approach I settled on was proxy labels. HCM causes the heart wall to thicken, which produces specific ECG changes — mainly increased voltages and hypertrophy patterns. Other conditions that also cause hypertrophy produce similar ECG signatures. So I used those conditions as HCM proxies: LVH (left ventricular hypertrophy), RVH (right ventricular hypertrophy), LVOLT (low voltage), and SEHYP (septal hypertrophy). The logic is that a model trained to detect these ECG patterns is learning the same features that would appear in true HCM.

6.2  Proxy Codes I Used
SCP Code
Condition
Why It's a Valid HCM Proxy
LVH
Left Ventricular Hypertrophy
Primary proxy — thickened LV wall produces same increased precordial voltages as HCM
RVH
Right Ventricular Hypertrophy
Similar hypertrophy-related voltage patterns
LVOLT
Low Voltage
Associated voltage abnormality pattern
SEHYP
Septal Hypertrophy
Closest to actual HCM — septal thickening is a defining HCM feature


6.3  Label Extraction Code — label_ecg.py
# label_ecg.py
import pandas as pd
import ast
 
BASE = 'datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
 
# Load the main metadata CSV
df = pd.read_csv(f'{BASE}/ptbxl_database.csv', index_col='ecg_id')
 
# The scp_codes column is stored as a string like "{'LVH': 100.0, 'NORM': 0.0}"
# ast.literal_eval converts it back to an actual Python dict
df.scp_codes = df.scp_codes.apply(ast.literal_eval)
 
# Load SCP code descriptions
scp = pd.read_csv(f'{BASE}/scp_statements.csv', index_col=0)
 
# These are the proxy codes I decided to use
HCM_CODES = {'LVH', 'RVH', 'LVOLT', 'SEHYP'}
 
def has_hcm_code(scp_dict):
    # Returns 1 if any HCM proxy code is present in this recording's labels
    return int(any(code in HCM_CODES for code in scp_dict.keys()))
 
df['hcm_label'] = df.scp_codes.apply(has_hcm_code)
 
print(df['hcm_label'].value_counts())
# Output: 0    19388
#         1     2449
 
df[['hcm_label']].to_csv('datasets/hcm_labels.csv')
print('Saved to datasets/hcm_labels.csv')

RESULT
19,388 control ECGs (label 0) and 2,449 HCM-proxy ECGs (label 1). Class imbalance: ~7.9 to 1. Labels saved.


NOTE
Using proxy labels is a real limitation and I want to be upfront about it. A proper clinical validation would need pathologist-confirmed HCM diagnoses linked to ECG records. That data doesn't exist publicly. Proxy labels are a reasonable compromise for a research prototype but they mean I can't make strong clinical claims about performance.


Section 7 — Model 1: Random Forest Baseline
DAY 4
Random Forest Baseline — train_baseline.py
ENTRY 7.1
DATE: 2/18/2026 


7.1  Why Start With a Baseline at All?
I wanted to start with something simple before going anywhere near deep learning. A baseline tells you the floor: what can you do with the most straightforward possible approach? Any complex model I build later needs to beat this, or the added complexity is pointless.
Random Forest was the obvious choice. It needs fixed-size feature vectors rather than raw waveforms, so I had to do some feature engineering first, but it handles class imbalance well with class_weight='balanced' and it is genuinely interpretable.

7.2  Feature Engineering
Random Forest can't process raw time-series, so I extracted 60 statistical features per ECG — 5 per lead, across all 12 leads:
Mean amplitude — average signal level
Standard deviation — how much the signal varies
Maximum amplitude — highest positive deflection (R-wave peak)
Minimum amplitude — deepest negative deflection (S-wave depth)
Amplitude range — max minus min, total voltage swing

7.3  Code — train_baseline.py
# train_baseline.py
# Result: AUC 0.8874 | Recall 0.28 | Precision 0.73 | F1 0.40
import pandas as pd, numpy as np, wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
 
BASE = 'datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
 
labels = pd.read_csv('datasets/hcm_labels.csv', index_col='ecg_id')
df = pd.read_csv(f'{BASE}/ptbxl_database.csv', index_col='ecg_id')
df['hcm_label'] = labels['hcm_label']
 
def extract_features(row):
    try:
        record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
        sig = record.p_signal  # (1000, 12)
        features = []
        for lead in range(12):
            ch = sig[:, lead]
            features += [np.mean(ch), np.std(ch), np.max(ch), np.min(ch),
                         np.max(ch) - np.min(ch)]
        return features
    except:
        return None
 
print('Extracting features (this takes ~5-10 min)...')
df['features'] = df.apply(extract_features, axis=1)
df = df.dropna(subset=['features'])
 
X = np.array(df['features'].tolist())
y = df['hcm_label'].values
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
 
# class_weight='balanced' adjusts weights inversely proportional to class frequency
clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)
 
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
 
print(classification_report(y_test, y_pred, target_names=['Control', 'HCM']))
print(f'AUC: {roc_auc_score(y_test, y_prob):.4f}')

RESULT
AUC: 0.8874 | HCM Recall: 0.28 | HCM Precision: 0.73 | HCM F1: 0.40 | Accuracy: 91%


7.4  What the Results Actually Mean
AUC of 0.8874 looks decent, but HCM Recall of 0.28 is a real problem. The model is only catching 28% of actual HCM cases. It misses 72% of the patients we care most about finding. High precision (0.73) just means that when it does flag someone, it's usually right — but it almost never flags anyone in the first place.
The model learned that predicting 'healthy' is almost always correct because 88% of the dataset is healthy. It is playing it safe at the expense of clinical utility. Statistical features are not enough — I needed to go to raw waveforms. Time to build a CNN.

Section 8 — Model 2: 1D Convolutional Neural Network
DAY 5
1D-CNN on Raw ECG Waveforms — train_cnn.py
ENTRY 8.1
DATE: 2/19/2026 to 2/20/2026


8.1  Why a CNN?
The Random Forest failed because statistical summaries throw away the shape of the signal. HCM shows up in specific waveform morphology — the exact height of the R-wave, the depth of the S-wave, whether the T-wave flips negative. None of that survives being collapsed into a mean or a standard deviation.
A 1D-CNN processes the raw time series directly. It applies learnable filters along the time axis that can detect waveform shapes — rising slopes, sharp peaks, inverted T-waves — without being explicitly programmed to look for them. The filters learn from the labeled data.

8.2  Architecture
Layer
Input
Output
What It Does
Conv1d(12→32, k=7)
(12, 1000)
(32, 1000)
Learns broad waveform features across all 12 leads
ReLU + MaxPool(2)
(32, 1000)
(32, 500)
Non-linearity + halves the time resolution
Conv1d(32→64, k=5)
(32, 500)
(64, 500)
Combines features from the first layer
ReLU + MaxPool(2)
(64, 500)
(64, 250)
Non-linearity + halves again
Conv1d(64→128, k=3)
(64, 250)
(128, 250)
High-level abstract features
ReLU + AdaptiveAvgPool
(128, 250)
(128, 1)
Collapses the time dimension
Flatten + Linear(128→64)
(128,)
(64,)
Classification head
Dropout(0.3) + Linear(64→1)
(64,)
(1,)
Output: HCM probability (logit)


8.3  Code — train_cnn.py
# train_cnn.py
# Result: AUC 0.8723 | Recall 0.78 | Precision 0.33 | F1 0.47
# NOTE: still using recording-level split here — leakage present
import pandas as pd, numpy as np, wfdb, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
 
BASE = 'datasets/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1'
 
class ECGDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows.reset_index()
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        try:
            record = wfdb.rdrecord(f"{BASE}/{row['filename_lr']}")
            sig = record.p_signal.T.astype(np.float32)  # shape (12, 1000)
            # Z-score normalize: mean 0, std 1 per recording
            sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        except:
            sig = np.zeros((12, 1000), dtype=np.float32)
        return torch.tensor(sig), torch.tensor(int(row['hcm_label']))
 
class ECG_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.net(x))
 
df = pd.read_csv(f'{BASE}/ptbxl_database.csv', index_col='ecg_id')
df['hcm_label'] = pd.read_csv('datasets/hcm_labels.csv', index_col='ecg_id')['hcm_label']
 
# Recording-level split — will fix this in V2
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['hcm_label'], random_state=42)
 
pos = (train_df['hcm_label'] == 1).sum()
neg = (train_df['hcm_label'] == 0).sum()
pos_weight = torch.tensor([neg / pos])  # ~7.9
 
train_loader = DataLoader(ECGDataset(train_df), batch_size=32, shuffle=True)
test_loader  = DataLoader(ECGDataset(test_df),  batch_size=32, shuffle=False)
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ECG_CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
 
for epoch in range(10):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.float().to(device)
        optimizer.zero_grad()
        loss = criterion(model(X).squeeze(), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/10 - Loss: {total_loss/len(train_loader):.4f}')
 
# Evaluate
model.eval()
all_probs, all_labels = [], []
with torch.no_grad():
    for X, y in test_loader:
        probs = torch.sigmoid(model(X.to(device))).cpu().squeeze().numpy()
        all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
        all_labels.extend(y.numpy())
 
all_probs, all_labels = np.array(all_probs), np.array(all_labels)
preds = (all_probs > 0.5).astype(int)
print(classification_report(all_labels, preds, target_names=['Control', 'HCM']))
print(f'AUC: {roc_auc_score(all_labels, all_probs):.4f}')
torch.save(model.state_dict(), 'hcm_cnn.pt')

RESULT
AUC: 0.8723 | HCM Recall: 0.78 | HCM Precision: 0.33 | HCM F1: 0.47 | Accuracy: 80%


8.4  What Changed and What's Still Broken
Recall jumped from 0.28 to 0.78 — the CNN is finding HCM cases the Random Forest completely missed. That confirms raw waveforms contain meaningful HCM signal that statistics alone can't capture. Good.
But precision cratered to 0.33. Two thirds of the patients the model flags as HCM are actually healthy. That is too many false alarms. Also, while reviewing the methodology I realized I still have a recording-level split — the same patient can have ECGs in both train and test. That is a data leakage problem. Both of these things get fixed in the next version.
NOTE
Identified during review: PTB-XL has multiple recordings per patient. A recording-level split lets the model see different ECGs from the same patient in both training and test sets. It may be learning individual cardiac signatures rather than generalizable HCM patterns. All future models will split by patient_id.


Section 9 — Model 3: Neuro-Symbolic V1 (Recording Split)
DAY 6
First Neuro-Symbolic Model — NS V1 (Recording Split)
ENTRY 9.1
DATE: 2/20/2026 to 2/23/2026


9.1  Adding the Symbolic Layer
This is the first version where I combined the CNN with actual clinical rules. The idea behind neuro-symbolic AI here is simple: the CNN learns patterns from the raw waveform that are hard to articulate, and the symbolic layer adds three explicit clinical criteria that cardiologists use every day. These two things get concatenated before the final classification layer, so the model can use both.
The reason I care about this: a pure CNN gives you a probability with no explanation. If a doctor asks 'why did you flag this patient?' the model has nothing to say. The symbolic rules make the model's reasoning auditable. It can flag someone and also say 'Sokolow-Lyon criterion exceeded, T-wave inversion present in V5' — something a clinician can actually evaluate.

9.2  The Symbolic Rules
def symbolic_score(sig):
    """
    Takes raw signal (1000, 12) BEFORE normalization.
    Normalization removes absolute voltage info — the rules need raw millivolt values.
    Returns 0.0, 0.33, 0.67, or 1.0 depending on how many rules are met.
    """
    score = 0
 
    # Rule 1: Sokolow-Lyon — S in V1 + R in V5 > 3.5 mV
    # Standard clinical LVH criterion. V1 is lead index 6, V5 is lead index 10.
    s_v1 = abs(np.min(sig[:, 6]))   # S-wave is the deepest negative deflection in V1
    r_v5 = np.max(sig[:, 10])        # R-wave is the tallest positive deflection in V5
    if (s_v1 + r_v5) > 3.5:
        score += 1
 
    # Rule 2: Cornell Voltage — S in V1 + R in aVL > 2.8 mV
    # Alternative LVH criterion. aVL is lead index 11.
    r_avl = np.max(sig[:, 11])
    if (s_v1 + r_avl) > 2.8:
        score += 1
 
    # Rule 3: T-wave inversion in V5
    # T-wave occupies roughly samples 600-800 in a 100Hz recording.
    t_wave_v5 = sig[600:800, 10]
    if np.min(t_wave_v5) < -0.3:   # threshold of -0.3 mV for clinically significant inversion
        score += 1
 
    return score / 3.0

9.3  The Fusion Architecture
class NeuroSymbolicECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(12, 32, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
            nn.Flatten()   # -> (batch, 128)
        )
        # 128 CNN features + 1 symbolic score = 129 total inputs to the FC layer
        self.fc = nn.Sequential(
            nn.Linear(129, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
 
    def forward(self, x, s):
        cnn_out = self.cnn(x)                      # (batch, 128)
        combined = torch.cat([cnn_out, s], dim=1)  # (batch, 129)
        return self.fc(combined)

RESULT
AUC: 0.9003 | Recall: 0.84 | Precision: 0.36 | F1: 0.51 — first time AUC crossed 0.90


NOTE
AUC 0.9003 looks great, but this version still uses recording-level splitting. These results are inflated by data leakage. I kept this model in the notebook for comparison purposes, but NS V2 is the first result I actually trust.


Section 10 — Model 4: Neuro-Symbolic V2 (Patient Split)
DAY 7
Patient-Level Split — First Scientifically Valid Result
ENTRY 10.1
DATE: 2/23/2026 to 2/25/2026


10.1  Why the Previous Split Was a Problem
PTB-XL has multiple ECG recordings per patient — up to 10. When I split recordings randomly, the same patient could have 6 recordings in training and 4 in the test set. The model then gets evaluated partly on ECGs from patients it has already seen. It might be learning individual cardiac signatures — 'this is patient 1234's heart' — rather than 'this is what HCM looks like in general.'
The fix is obvious once you see the issue: split by patient ID, not by recording. If a patient is in the test set, none of their recordings appear anywhere in training or validation. This is the standard in published clinical AI research. The AUC drop from NS V1 to NS V2 (0.9003 to 0.8874) directly measures how much the leakage was inflating the previous result.

10.2  Patient-Level Split Code
# The key change: group by patient_id before splitting
 
unique_patients = df['patient_id'].unique()
np.random.seed(42)
np.random.shuffle(unique_patients)
 
n = len(unique_patients)
# 70% train / 15% validation / 15% test — by patient, not recording
train_patients = set(unique_patients[:int(0.70 * n)])
val_patients   = set(unique_patients[int(0.70 * n):int(0.85 * n)])
test_patients  = set(unique_patients[int(0.85 * n):])
 
train_df = df[df['patient_id'].isin(train_patients)]
val_df   = df[df['patient_id'].isin(val_patients)]
test_df  = df[df['patient_id'].isin(test_patients)]
 
# Balance the training set only
# Val and test stay at the real clinical distribution (88/12 split)
hcm_train  = train_df[train_df['hcm_label'] == 1].sample(n=min(2000, ...), random_state=42)
ctrl_train = train_df[train_df['hcm_label'] == 0].sample(n=min(2000, ...), random_state=42)
train_df   = pd.concat([hcm_train, ctrl_train]).sample(frac=1, random_state=42)

10.3  Early Stopping and Validation Monitoring
I also added two improvements that should have been in V1:
best_val_auc = 0
patience = 3      # stop if val AUC doesn't improve for 3 consecutive epochs
no_improve = 0
 
for epoch in range(20):
    # ... training ...
 
    # Check val AUC after every epoch
    val_auc = roc_auc_score(val_labels, val_probs)
    print(f'Epoch {epoch+1} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}')
 
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), 'hcm_neurosymbolic_v2.pt')  # save the best checkpoint
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
 
# Always load the best checkpoint, not the final epoch
model.load_state_dict(torch.load('hcm_neurosymbolic_v2.pt'))

RESULT
AUC: 0.8874 | Recall: 0.77 | Precision: 0.41 | F1: 0.53 — first result I actually trust


The 0.013 AUC drop from V1 to V2 is not a failure. It is exactly the measurement I needed to quantify how much data leakage was inflating the previous results. This is now the honest baseline to improve from.

Section 11 — Model 5: Neuro-Symbolic V3 (Focal Loss)
DAY 8
Focal Loss — Fixing the False Positive Problem
ENTRY 11.1
DATE: 2/25/2026 to 2/26/2026


11.1  The Problem I Was Trying to Fix
V2 has precision of 0.41. That means 59% of the patients the model flags as HCM are actually healthy. In a real clinical setting, every false positive means an unnecessary echocardiogram, a scared patient, and wasted resources. I needed the model to be more selective — only flag someone when it's reasonably confident.
The root cause is that binary cross-entropy loss treats all mistakes equally. With 88% of the data being healthy ECGs, the model sees an enormous number of easy 'true negative' predictions where it correctly identifies healthy patients. These swamp the gradient updates and the model learns to just predict healthy, because that's almost always right.

11.2  Focal Loss — What It Does
Focal Loss (Lin et al., 2017) solves this by multiplying the standard loss by a factor that shrinks to nearly zero when the model is confident and correct, and stays near one when the model is wrong or uncertain. Easy examples contribute almost nothing to the gradient. Hard borderline cases — the subtle HCM ECGs that look nearly normal — dominate training. That's exactly what I needed.
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        """
        alpha: class weighting factor (0.25 works well here)
        gamma: focusing strength. 0 = standard BCE. 2.0 = Lin et al. recommendation.
               Higher values = more focus on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
 
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        pt = torch.exp(-bce)   # model confidence in correct prediction
 
        # (1 - pt)^gamma: near 0 when confident, near 1 when uncertain
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()
 
# Replace BCEWithLogitsLoss with this:
criterion = FocalLoss(alpha=0.25, gamma=2.0)

11.3  Threshold Analysis
I also stopped using the default 0.5 cutoff and actually looked at how performance changes across thresholds. This showed the real operating tradeoffs:
Threshold
Precision
Recall
F1
Clinical Use Case
0.30
0.13
0.99
0.23
Catch everything — mass screening, lots of false alarms
0.40
0.16
0.98
0.28
High sensitivity mode
0.50
0.25
0.90
0.40
Default — not great for either goal
0.60
0.46
0.79
0.59
Better balance for screening
0.70
0.66
0.61
0.63
Confirmatory — only flag when fairly confident


RESULT
AUC: 0.8773 | Precision: 0.57 | Recall: 0.68 | F1: 0.63 at threshold 0.70 — precision jumped from 0.41 to 0.57


Focal loss directly did what it was supposed to do. Precision up 16 points, F1 hits 0.63 for the first time. AUC dipped slightly but the precision-recall balance improved meaningfully. I'll take that trade.

Section 12 — Model 6: Neuro-Symbolic V4 (Residual CNN)
DAY 9
Residual Connections — Going Deeper Without Degradation
ENTRY 12.1
DATE: 2/26/2026 to 2/27/2026


12.1  Why I Wanted to Go Deeper
The CNN in previous versions was 3 convolutional layers. I suspected there were more complex ECG patterns the model could learn with a deeper architecture. But when I tried just adding more conv layers, performance didn't improve — it actually got slightly worse. The reason is vanishing gradients.
During backpropagation, error signals travel backwards through every layer as a chain of multiplied values. With enough layers, these products shrink exponentially. By the time the gradient reaches the early layers, it is essentially zero. Those layers stop learning. The network is theoretically deep but practically shallow.

12.2  Residual Connections (He et al., 2016)
The ResNet paper solved this elegantly. Instead of learning a transformation F(x), learn F(x) + x, where x is the input passed directly through a shortcut connection that bypasses the layers. When gradients flow back, they can take the shortcut and bypass the chain of multiplications entirely. Early layers get real gradient signal and actually learn. This enabled networks 50-150 layers deep. I used the same idea for my 1D ECG CNN.
class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),   # normalize activations across the batch
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()
 
    def forward(self, x):
        # Shortcut: output = F(x) + x
        # Gradient during backprop flows back through the shortcut directly
        return self.relu(self.block(x) + x)
 
class NeuroSymbolicECG_V4(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(12, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2)
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(64), nn.MaxPool1d(2),
            ResidualBlock(64), nn.MaxPool1d(2),
        )
        self.output_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(65, 64), nn.ReLU(), nn.Dropout(0.4),  # 64 CNN + 1 symbolic
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x, s):
        x = self.input_conv(x)
        x = self.res_blocks(x)
        x = self.flatten(self.output_pool(x))
        return self.fc(torch.cat([x, s], dim=1))

12.3  Expanded Symbolic Rules
I also extended the symbolic layer from 3 rules to 5:
def symbolic_score(sig):
    score = 0
    s_v1  = abs(np.min(sig[:, 6]))
    r_v5  = np.max(sig[:, 10])
    r_avl = np.max(sig[:, 11])
 
    if (s_v1 + r_v5) > 3.5: score += 1   # Sokolow-Lyon
    if (s_v1 + r_avl) > 2.8: score += 1  # Cornell voltage
    if np.min(sig[600:800, 10]) < -0.3: score += 1  # T-wave inversion V5
 
    # Rule 4 (new): sum of peak voltages across all 12 leads
    # Diffuse hypertrophy shows up as high total voltage
    voltage_sum = np.sum([np.max(np.abs(sig[:, l])) for l in range(12)])
    if voltage_sum > 15.0: score += 1
 
    # Rule 5 (new): R-wave amplitude in aVL alone
    # Isolated R in aVL > 1.1 mV is an independent LVH criterion
    if r_avl > 1.1: score += 1
 
    return score / 5.0

I also added a cosine annealing learning rate scheduler — the LR decreases following a cosine curve rather than staying flat, which helps avoid oscillating around local minima late in training.

RESULT
AUC: 0.9076 (best of all models) | Precision: 0.59 | Recall: 0.68 | F1: 0.63


Section 13 — Model 7: Neuro-Symbolic V5 (Final Model)
DAY 10
Attention + Augmentation + Auto Threshold — FINAL
ENTRY 13.1
DATE: 2/27/2026 to 3/1/2026


13.1  What I Added in the Final Version
Attention pooling — instead of averaging all ECG time steps equally, the model learns which segments matter most for HCM detection
ECG augmentation — random noise, amplitude scaling, and time shifts during training to improve generalization
Label smoothing — prevents the model from becoming overconfident in its predictions
Automatic threshold selection — find the best threshold on the validation set rather than defaulting to 0.5

13.2  Attention Pooling
The AdaptiveAvgPool in previous versions treated every time step of the ECG the same. But for HCM, some parts of the ECG matter far more than others — the QRS complex voltage, the T-wave morphology, the ST segment. Attention pooling learns a scalar weight for each time step during training. Steps with HCM-relevant features get higher weights; uninformative segments get downweighted.
class AttentionPool1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),  # 1x1 conv: one weight per timestep
            nn.Softmax(dim=2)                       # weights sum to 1 across time
        )
 
    def forward(self, x):
        # x shape: (batch, channels, time)
        weights = self.attn(x)           # (batch, 1, time)
        return (x * weights).sum(dim=2)  # weighted sum -> (batch, channels)
 
# In the V5 model — swapped in for AdaptiveAvgPool1d:
self.attention = AttentionPool1D(64)

13.3  ECG Augmentation
Augmentation is a standard trick for improving generalization. Instead of seeing the exact same ECG every epoch, the training set gets slightly randomized versions of each recording. This forces the model to learn the actual pattern rather than memorizing specific recordings.
def augment_ecg(sig_t):
    # Only applied during training, never during eval
 
    # Random Gaussian noise — mimics electrode contact noise
    if np.random.random() < 0.5:
        sig_t = sig_t + np.random.normal(0, 0.02, sig_t.shape).astype(np.float32)
 
    # Amplitude scaling — mimics patient body habitus variation
    # Kept in 0.85-1.15 range so Sokolow-Lyon thresholds still roughly hold
    if np.random.random() < 0.5:
        sig_t = sig_t * np.random.uniform(0.85, 1.15)
 
    # Circular time shift — mimics recording start point variation
    if np.random.random() < 0.3:
        sig_t = np.roll(sig_t, np.random.randint(-50, 50), axis=1)
 
    return sig_t

13.4  Automatic Threshold Selection
In all previous models I just used whatever threshold gave the best F1 from the threshold analysis. In V5 I automated this properly: sweep thresholds on the validation set, pick the best one, then apply it to the test set. The validation set is entirely separate from the test set so there is no information leakage.
def find_optimal_threshold(val_probs, val_labels):
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.3, 0.8, 0.05):
        preds = (val_probs > thresh).astype(int)
        p = precision_score(val_labels, preds, zero_division=0)
        r = recall_score(val_labels, preds)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh
 
# Find on val set, apply to test set:
optimal_thresh = find_optimal_threshold(val_probs, val_labels)
# -> 0.70
preds = (test_probs > optimal_thresh).astype(int)

RESULT
AUC: 0.9043 | Precision: 0.66 (highest of all models) | Recall: 0.61 | F1: 0.63 | Accuracy: 92% | Threshold: 0.70


NOTE
Val AUC 0.9121 vs test AUC 0.9043 — gap of 0.0078 is narrow. Good generalization. Early stopping triggered at epoch 13 out of 20 maximum.


Section 14 — Visualization Generation
DAY 11
generate_visualizations.py — Poster Charts
ENTRY 14.1
DATE: 3/1/2026 to 3/2/2026


14.1  Charts I Generated
Panel 1 — ROC curves for all 7 models on one plot with AUC values
Panel 2 — AUC progression bar chart with the 0.90 target line
Panel 3 — HCM F1 score trajectory across all models
Panel 4 — Confusion matrix for NS V5 at threshold 0.70
Panel 5 — Precision / Recall / F1 vs threshold sweep for NS V5
Panel 6 — Side-by-side grouped metrics for all models
Standalone — Accuracy comparison with the 88% naive baseline marked

14.2  Confusion Matrix — NS V5 at Threshold 0.70


Predicted: Control
Predicted: HCM
Actual: Control
2,772  (True Negatives)
120  (False Positives)
Actual: HCM
149  (False Negatives)
231  (True Positives)


Total test patients: 3,272. The model correctly identifies 2,772 healthy patients and 231 HCM patients. It incorrectly flags 120 healthy patients — those will get unnecessary follow-up. It misses 149 HCM patients — those are the most clinically dangerous errors. Threshold 0.70 was the optimal tradeoff the automatic selection procedure found on the validation set.

14.3  Run Command
python generate_visualizations.py
# Outputs:
# hcm_results_poster.png       18x12 in, 150 DPI, 6-panel figure
# hcm_accuracy_comparison.png  11x6 in, accuracy comparison with baseline

Section 15 — Results Analysis and Interpretation
ENTRY 15.1
DATE: 3/2/2026 to 3/3/2026


15.1  All Model Results
Model
AUC
F1
Precision
Recall
Accuracy
Valid?
Random Forest
0.8874
0.40
0.73
0.28
91%
Baseline
1D-CNN
0.8723
0.47
0.33
0.78
80%
Rec-split
NS V1
0.9003
0.51
0.36
0.84
82%
Inflated
NS V2
0.8874
0.53
0.41
0.77
84%
First valid
NS V3
0.8773
0.63
0.57
0.68
84%
Valid
NS V4
0.9076
0.63
0.59
0.68
65%
Valid
NS V5 Final
0.9043
0.63
0.66
0.61
92%
Valid


15.2  The Story Each Iteration Tells
The numbers only mean something in context. Here's what each version was actually solving:

RF to CNN: Statistical features discard waveform shape. Switching to raw signal processing jumps recall from 0.28 to 0.78.
CNN to NS V1: Pure deep learning is uninterpretable. Adding clinical rules crosses AUC 0.90 and gives the model explainable reasoning.
NS V1 to NS V2: Recording-level splitting inflates metrics. Fixing to patient-level splitting drops AUC by 0.013 — that difference is the exact size of the data leakage.
NS V2 to NS V3: Standard BCE loss ignores hard cases in imbalanced data. Focal loss jumps precision from 0.41 to 0.57, F1 hits 0.63 for the first time.
NS V3 to NS V4: Shallow CNN caps representational capacity. Residual connections enable a deeper network, best AUC of all models at 0.9076.
NS V4 to NS V5: Global average pooling ignores which ECG segments are diagnostically relevant. Attention pooling focuses on them. Highest precision at 0.66.

15.3  Why AUC is the Right Primary Metric
A model that always predicts 'healthy' achieves 88% accuracy — that is the naive baseline. Accuracy is completely uninformative here. AUC measures the model's ability to rank HCM patients above healthy patients regardless of threshold. It is the metric used in clinical AI publications for exactly this reason. 0.9043 is well above the clinical usefulness threshold and above the 0.85 target I set at the start.

Section 16 — Limitations and Future Work
ENTRY 16.1
DATE: 3/4/2026


16.1  What This Project Can't Claim
Proxy Labels
This is the biggest one. PTB-XL doesn't have confirmed HCM diagnoses. I used LVH and related codes as proxies because they produce similar ECG patterns, but they are not HCM. The model learned to detect hypertrophy-related ECG changes broadly — not HCM specifically. A paper with actual pathologist-confirmed HCM labels linked to ECG records would be needed to make any real clinical performance claims. That data doesn't exist publicly.

Training Sample Size
RAM constraints on my laptop capped training at 2,000 samples per class — 4,000 total out of a possible 21,837. I ran the full dataset size in early experiments and my laptop froze. Training on all available data with proper patient-level splitting would require Google Colab or AWS. Based on how AUC scaled with sample size in early runs, I'd expect to push past 0.92.

Not a Diagnostic Device
This is a research prototype. It was not clinically validated, has not been tested on external datasets, and has not been reviewed by any regulatory body. It cannot replace echocardiography or a cardiologist. Calling this a 'diagnostic tool' would be inaccurate. It is a proof of concept for the approach.

100 Hz Only
PTB-XL has 500 Hz recordings too — five times the temporal resolution. Higher resolution captures faster waveform features (high-frequency QRS notches, precise slope changes) that might be HCM-relevant. I couldn't use them because the file sizes were too large for my setup.

16.2  What I'd Do Next
Train on all 21,837 recordings using cloud compute. Patient-level splitting throughout. Expect meaningful AUC improvement.
Find a clinical partner to get real HCM-labeled ECGs for validation. This would change the project from a proof of concept into something clinically testable.
Add Grad-CAM saliency maps adapted for 1D CNN. This would visualize which specific ECG segments the model uses for each prediction — something clinicians could actually look at.
Use the 500 Hz recordings. More temporal resolution might catch features invisible at 100 Hz.
Test on external ECG datasets to check generalization beyond PTB-XL.
Build a simple web demo interface. Even a basic UI would make the prototype much easier to demonstrate.

Section 17 — Project Summary and Conclusions
ENTRY 17.1
DATE: 3/5/2026


17.1  Did I Meet the Goals I Set?
Goal
Target
Result
Status
AUC on patient-level test set
> 0.85
0.9043
Met
Interpretability via clinical rules
Required
5 symbolic rules fused with CNN output
Met
CPU-only training and inference
Required
All models run on CPU
Met
Patient-level methodology
Required
Patient splits from V2 onward
Met
Reproducible results
Required
Random seed 42 throughout
Met
At least 5 model versions
5 versions
7 versions with documented improvements
Met


17.2  What I Think Actually Mattered
A few things made the difference in this project. The neuro-symbolic architecture improved precision over the pure CNN significantly — 0.66 vs 0.33 — while keeping competitive AUC. That is the core contribution. But the methodology fix from recording-level to patient-level splitting was equally important. Without that, I would have been reporting inflated numbers and not known it.
Focal loss was probably the single biggest performance improvement in terms of F1 — jumping precision from 0.41 to 0.57 in one change. That came from reading a paper and understanding why the standard approach was failing for this class of problem.

17.3  Final Thoughts
This started as a question about why athletes die suddenly. Seven model versions later it ended with a working AI system achieving AUC 0.9043 on completely unseen patients, running on a laptop, with decisions that connect to the same rules a cardiologist would use. The limitations are real — proxy labels, constrained training set, no clinical validation. I've tried to be honest about all of those.
What I can say: the approach works, the methodology is sound, and the engineering decisions were each made for a documented reason with a measurable outcome. That's what the notebook is for.

FINAL RESULTS — NS V5
AUC: 0.9043     HCM F1: 0.63     Precision: 0.66     Recall: 0.61
Accuracy: 92%   Threshold: 0.70   Val AUC: 0.9121    Test AUC: 0.9043
Dataset: PTB-XL   21,837 ECGs   Patient-level split   CPU-only


References
All sources cited during this project, in APA 7th edition format.
[1]  He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 770–778). IEEE. https://doi.org/10.1109/CVPR.2016.90
[2]  Ko, W. Y., Siontis, K. C., Attia, Z. I., Carter, R. E., Kapa, S., Cha, S. S., Asirvatham, S. J., Friedman, P. A., & Noseworthy, P. A. (2020). Detection of hypertrophic cardiomyopathy using a convolutional neural network-enabled electrocardiogram. Journal of the American College of Cardiology, 75(7), 722–733. https://doi.org/10.1016/j.jacc.2019.12.030
[3]  Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) (pp. 2980–2988). IEEE. https://doi.org/10.1109/ICCV.2017.324
[4]  Maron, B. J., Gardin, J. M., Flack, J. M., Gidding, S. S., Kurosaki, T. T., & Bild, D. E. (1995). Prevalence of hypertrophic cardiomyopathy in a general population of young adults: Echocardiographic analysis of 4111 subjects in the CARDIA study. Circulation, 92(4), 785–789. https://doi.org/10.1161/01.CIR.92.4.785
[5]  Maron, B. J., Haas, T. S., Murphy, C. J., Ahluwalia, A., & Rutten-Ramos, S. (2014). Incidence and causes of sudden death in U.S. college athletes. Journal of the American College of Cardiology, 63(16), 1636–1643. https://doi.org/10.1016/j.jacc.2014.01.041
[6]  Maron, B. J., & Maron, M. S. (2013). Hypertrophic cardiomyopathy. The Lancet, 381(9862), 242–255. https://doi.org/10.1016/S0140-6736(12)60397-3
[7]  Ommen, S. R., Mital, S., Burke, M. A., Day, S. M., Deswal, A., Elliott, P., Evanovich, L. L., Hung, J., Joglar, J. A., Kantor, P., Kimmelstiel, C., Kittleson, M., Link, M. S., Maron, M. S., Martinez, M. W., Miyake, C. Y., Schaff, H. V., Semsarian, C., & Towbin, J. A. (2020). 2020 AHA/ACC guideline for the diagnosis and treatment of patients with hypertrophic cardiomyopathy. Journal of the American College of Cardiology, 76(25), e159–e240. https://doi.org/10.1016/j.jacc.2020.08.045
[8]  Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Kopf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS) (Vol. 32, pp. 8024–8035). Curran Associates. https://arxiv.org/abs/1912.01703
[9]  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
[10]  Siontis, K. C., Noseworthy, P. A., Attia, Z. I., & Friedman, P. A. (2021). Artificial intelligence-enhanced electrocardiography in cardiovascular disease management. Nature Reviews Cardiology, 18(7), 465–478. https://doi.org/10.1038/s41569-020-00503-2
[11]  Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning for ECG analysis: Benchmarks and insights from PTB-XL. IEEE Journal of Biomedical and Health Informatics, 25(5), 1519–1528. https://doi.org/10.1109/JBHI.2020.3022989
[12]  Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS) (Vol. 30, pp. 5998–6008). Curran Associates. https://arxiv.org/abs/1706.03762
[13]  Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. Scientific Data, 7, Article 154. https://doi.org/10.1038/s41597-020-0495-6
[14]  Wightman, G., Maron, B. J., & Ellison, R. C. (1985). Sokolow-Lyon criteria and echocardiographic left ventricular hypertrophy. The American Journal of Cardiology, 55(12), 1749–1751. https://doi.org/10.1016/0002-9149(85)90543-0
[15]  Xu, W., Zhu, F., Zhang, Y., Li, P., & Sheng, Y. (2024). An overview of the treatments for hypertrophic cardiomyopathy. Frontiers in Cardiovascular Medicine, 11, Article 1387596. https://doi.org/10.3389/fcvm.2024.1387596
[16]  Zhou, B., Khosla, A., Lapedriza, A., Oliva, A., & Torralba, A. (2016). Learning deep features for discriminative localization (Grad-CAM). In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 2921–2929). IEEE. https://doi.org/10.1109/CVPR.2016.319 
[17] Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

Link to git repository:  https://github.com/satkatra/Hypertrophic-cardiomyopathy-neuro-symbolic-AI-model-




