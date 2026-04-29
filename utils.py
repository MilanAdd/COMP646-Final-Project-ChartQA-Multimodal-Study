import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config

plt.rcParams.update({
    "font.family":"serif",
    "font.size":11,
    "axes.titlesize":12,
    "axes.labelsize":11,
    "xtick.labelsize":9,
    "ytick.labelsize":9,
    "legend.fontsize":9,
    "figure.dpi":150,
    "axes.spines.top":False,
    "axes.spines.right":False,
    "axes.grid":True,
    "grid.alpha":0.3,
    "grid.linestyle":"--"
})

COLORS = {"frozen":"#2E86AB","lora":"#E67E22","zeroshot":"#27AE60"}

LABELS = {"frozen":"Frozen CLIP","lora":"LoRA CLIP","zeroshot":"Zero-shot Qwen2.5-VL"}

def load_history(mode:str) -> dict:
    path = os.path.join(config.RESULTS_DIR,f"history_{mode}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"History file not found: {path}")
    with open(path) as f:
        return json.load(f)

def load_eval(mode:str,split:str="test") -> dict:
    path = os.path.join(config.RESULTS_DIR,f"eval_{mode}_{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Eval file not found: {path}")
    with open(path) as f:
        return json.load(f)
    
def plot_training_curves(mode:str,save:bool=True) -> None:
    history = load_history(mode)
    epochs = list(range(1,len(history["train_loss"])+1))
    fig,(ax_loss,ax_acc) = plt.subplots(1,2,figsize=(10,4))
    color = COLORS[mode]

    ax_loss.plot(epochs,history["train_loss"],label="Train",color=color,linewidth=1.8)
    ax_loss.plot(epochs,history["val_loss"],label="Validation",color=color,linewidth=1.8,linestyle="--")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(f"Loss - {LABELS[mode]}")
    ax_loss.legend()
    ax_loss.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_acc.plot(epochs,history["train_acc"],label="Train",color=color,linewidth=1.8)
    ax_acc.plot(epochs,history["val_acc"],label="Validation",color=color,linewidth=1.8,linestyle="--")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title(f"Accuracy - {LABELS[mode]}")
    ax_acc.set_ylim(0,1)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.legend()
    ax_acc.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,f"training_curves_{mode}.pdf")
        plt.savefig(path,format="pdf",bbox_inches="tight")
        print(f"[Plot] Saved to {path}")

    plt.close()


def plot_accuracy_comparison(split:str="test",save:bool=True) -> None:
    modes = ["frozen","lora","zeroshot"]
    accs = []
    labels = []

    for mode in modes:
        try:
            data = load_eval(mode,split)
            acc = data["breakdowns"]["overall"]["accuracy"]
            accs.append(acc*100)
            labels.append(LABELS[mode])
        except FileNotFoundError:
            print(f"[Plot] Skipping {mode} - eval file not found")
    
    if not accs:
        print("[Plot] No eval files found for accuracy comparison")
        return
    
    fig,ax = plt.subplots(figsize=(7,4))
    colors = [COLORS[m] for m in modes[:len(accs)]]
    bars = ax.bar(labels,accs,color=colors,width=0.5,edgecolor="white")

    for bar,acc in zip(bars,accs):
        ax.text(bar.get_x() + bar.get_width()/2,bar.get_height()+0.5,f"{acc:.1f}%",ha="center",va="bottom",fontsize=10,fontweight="bold")
    
    ax.set_ylabel("Relaxed Accuracy (%)")
    ax.set_title(f"Overall Accuracy on ChartQA {split.capitalize()} Set")
    ax.set_ylim(0,max(accs)*1.15)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,"accuracy_comparison.pdf")
        plt.savefig(path,format="pdf",bbox_inches="tight")
        print(f"[Plot] Saved to {path}")
    
    plt.close()

def plot_breakdown_by_type(breakdown_key:str,split:str="test",save:bool=True) -> None:
    modes = ["frozen","lora","zeroshot"]
    all_data = {}

    for mode in modes:
        try:
            data = load_eval(mode,split)
            all_data[mode]= data["breakdowns"][breakdown_key]
        except FileNotFoundError:
            pass
    
    if not all_data:
        print(f"[Plot] No data found for {breakdown_key}")
        return
    
    categories = sorted(set(cat for d in all_data.values() for cat in d.keys()))
    n_cats = len(categories)
    n_modes = len(all_data)
    width = 0.8/n_modes
    x = np.arange(n_cats)

    fig,ax = plt.subplots(figsize=(max(7,n_cats*1.8),4.5))

    for i,(mode,breakdown) in enumerate(all_data.items()):
        accs = [breakdown.get(c,{}).get("accuracy",0)*100 for c in categories]
        offset = (i-n_modes/2 + 0.5) * width
        bars = ax.bar(x+offset,accs,width=width,label=LABELS[mode],color=COLORS[mode],edgecolor="white")

        for bar,acc in zip(bars,accs):
            if acc > 3:
                ax.text(bar.get_x() + bar.get_width()/2,bar.get_height()+0.4,f"{acc:.0f}%",ha="center",va="bottom",fontsize=7)
    
    clean_key = breakdown_key.replace("by_","").replace("_"," ").title()
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_"," ").title() for c in categories],rotation=15,ha="right")
    ax.set_ylabel("Relaxed Accuracy (%)")
    ax.set_xlabel(clean_key)
    ax.set_title(f"Accuracy by {clean_key} - ChartQA {split.capitalize()} Set")
    ax.set_ylim(0,100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save:
        fname = f"breakdown_{breakdown_key}_{split}.pdf"
        path = os.path.join(config.FIGURES_DIR,fname)
        plt.savefig(path,format="pdf",bbox_inches="tight")
        print(f"[Plot] Saved to {path}")
    
    plt.close()

def plot_breakdown_by_answer_type(split:str="test",save:bool=True)-> None:
    modes = ["frozen","lora","zeroshot"]
    all_data = {}

    for mode in modes:
        try:
            data = load_eval(mode,split)
            brkdwn = data["breakdowns"].get("by_answer_type",{})
            if brkdwn:
                all_data[mode] = brkdwn
        except:
            pass
    
    if not all_data:
        print("[Plot] No answer type data found. Rerun evals with updated evaluate.py script")
        return

    cats = ["binary","numerical","textual"]
    n_cats = len(cats)
    n_modes = len(all_data)
    width = 0.8/n_modes
    x = np.arange(n_cats)

    fig,ax = plt.subplots(figsize=(7,4.5))

    for i,(mode,breakdown) in enumerate(all_data.items()):
        accs = [breakdown.get(cat,{}).get("accuracy",0)*100 for cat in cats]
        offset = (i-n_modes/2+0.5)*width
        bars = ax.bar(x+offset,accs,width=width,label=LABELS[mode],color=COLORS[mode],edgecolor="white")
        for bar,acc in zip(bars,accs):
            if acc > 2:
                ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.5,f"{acc:0f}%",ha="center",va="bottom",fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["Binary","Numerical","Textual"])
    ax.set_ylabel("Relaxed Accuracy (%)")
    ax.set_xlabel("Answer Type")
    ax.set_title(f"Accuracy by Answer Type = ChartQA {split.capitalize()} Set")
    ax.set_ylim(0,100)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,f"breakdown_by_answer_type_{split}.pdf")
        plt.savefig(path,format="pdf",bbox_inches="tight")
        print(f"[Plot] Saved to {path}")
    plt.close()

def plot_cross_table(mode:str,split:str="test",save:bool=True) -> None:
    try:
        data = load_eval(mode,split)
    except FileNotFoundError:
        print(f"[Plot] No eval file for {mode} {split}")
        return
    
    cross = data["breakdowns"]["by_cross"]

    qtypes = sorted(set(k.split("__")[0] for k in cross.keys()))
    ctypes = sorted(set(k.split("__")[1] for k in cross.keys() if k.split("__")[1] != "unknown"))

    matrix = np.zeros((len(qtypes),len(ctypes)))
    counts = np.zeros((len(qtypes),len(ctypes)),dtype=int)

    for i,qt in enumerate(qtypes):
        for j,ct in enumerate(ctypes):
            key = f"{qt}__{ct}"
            stats = cross.get(key,{})
            matrix[i,j] = stats.get("accuracy",0) * 100
            counts[i,j] = stats.get("total",0)
    
    fig,ax = plt.subplots(figsize=(max(5,len(ctypes) * 1.5),max(3,len(qtypes)*1.2)))
    im = ax.imshow(matrix,cmap="Blues",vmin=0,vmax=100,aspect="auto")

    for i in range(len(qtypes)):
        for j in range(len(ctypes)):
            acc = matrix[i,j]
            n = counts[i,j]
            color = "white" if acc > 60 else "black"
            ax.text(j,i,f"{acc:.0f}%\n(n={n})",ha="center",va="center",fontsize=8,color=color)
    
    ax.set_xticks(range(len(ctypes)))
    ax.set_yticks(range(len(qtypes)))
    ax.set_xticklabels([c.replace("_"," ").title() for c in ctypes])
    ax.set_yticklabels([q.title() for q in qtypes])
    ax.set_xlabel("Chart Type")
    ax.set_ylabel("Question Type")
    ax.set_title(f"Accuracy Heatmap ({LABELS[mode]}) - "
                 f"ChartQA {split.capitalize()} Set")
    
    cbar = plt.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cbar.set_label("Relaxed Accuracy (%)")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    
    plt.tight_layout()

    if save:
        path = os.path.join(config.FIGURES_DIR,f"cross_table_{mode}_{split}.pdf")
        plt.savefig(path,format="pdf",bbox_inches="tight")
        print(f"[Plot] Saved to {path}")

    plt.close()


def print_latex_table(split:str="test") -> None:
    modes = ["frozen","lora","zeroshot"]

    header = (
        r"\begin{table}[h]" + "\n"
        r"  \centering" + "\n"
        r"  \caption{Relaxed accuracy (\%) on ChartQA test set by model variant, "
        r"question type, and chart type.}" + "\n"
        r"  \label{tab:results}" + "\n"
        r"  \begin{tabular}{lccccccc}" + "\n"
        r"  \toprule" + "\n"
        r"  Model & Overall & Human & Augmented & Bar & Line & Pie \\" + "\n"
        r"  \midrule"
    )

    print(header)

    for mode in modes:
        try:
            data = load_eval(mode,split)
            bd = data["breakdowns"]
            ov = bd["overall"]["accuracy"] * 100
            human = bd["by_question_type"].get("human",    {}).get("accuracy",0) * 100
            aug = bd["by_question_type"].get("augmented",    {}).get("accuracy",0) * 100
            bar = bd["by_chart_type"].get("bar", {}).get("accuracy",0) * 100
            line = bd["by_chart_type"].get("line", {}).get("accuracy",0) * 100
            pie = bd["by_chart_type"].get("pie", {}).get("accuracy",0) * 100
            label = LABELS[mode]

            print(f"  {label} & {ov:.1f} & {human:.1f} & {aug:.1f} "
                  f"& {bar:.1f} & {line:.1f} & {pie:.1f} \\\\")
            
        except FileNotFoundError:
            print(f"  % {mode} - eval file not found")
        
    
    footer = (
        r"  \bottomrule" + "\n"
        r"  \end{tabular}" + "\n"
        r"\end{table}"
    )
    print(footer)

if __name__ == "__main__":
    print("[Utils] Generating all report figures...\n")

    for mode in ("frozen","lora"):
        try:
            plot_training_curves(mode)
        except FileNotFoundError as e:
            print(f"[Utils] Skipping training curves for {mode}: {e}")
    
    plot_accuracy_comparison()

    plot_breakdown_by_type("by_question_type")
    plot_breakdown_by_type("by_chart_type")
    plot_breakdown_by_answer_type()


    for mode in ("frozen","lora","zeroshot"):
        try:
            plot_cross_table(mode)
        except FileNotFoundError as e:
            print(f"[Utils] Skipping cross table for {mode}: {e}")
    
    print("\n[Utils] LaTeX results table:\n")
    print_latex_table()

    print("\n[Utils] Done. Figures saved to:",config.FIGURES_DIR)
