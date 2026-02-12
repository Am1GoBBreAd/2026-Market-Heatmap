import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Data Dictionary
data = {
    'Institution': ['BofA', 'Goldman Sachs', 'UBS', 'Morgan Stanley', 'J.P. Morgan', 
                    'Macquarie', 'Barclays', 'DBS Bank', 'HSBC', 'Standard Chartered', 'BlackRock'],
    'US Equities': [3, 3, 3, 3, 3, 1, 1, 3, 3, 3, 3],
    'Europe': [2, 3, 3, 2, 3, 3, 3, 3, 2, 1, 3],
    'EM/Asia': [2, 3, 3, 2, 3, 1, 2, 3, 3, 3, 2],
    'China': [1, 2, 3, 3, 2, 3, 3, 3, 3, 3, 2],
    'Japan': [2, 3, 3, 2, 3, 3, 2, 3, 3, 1, 3]
}

# 2. Create DataFrame
df = pd.DataFrame(data).set_index('Institution')

# 3. Plotting
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, cmap='RdYlGn', linewidths=0.5)
plt.title('2026 Global Market Outlook Heatmap')

# For Notebook preview
plt.show() 

# For Script preview (saves to your file explorer)
plt.savefig('heatmap_preview.png')
