"""Static scientific figure for the paired catch evaluation."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

p=argparse.ArgumentParser();p.add_argument('report',type=Path);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
report=json.loads(args.report.read_text());run=Path(report['run_dir']);records=[json.loads(s) for s in (run/'diagnostic_episodes.jsonl').read_text().splitlines()]
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['axes.unicode_minus']=False
fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),dpi=150)
modes=['mean','noise_learned','noise_005'];labels=['无噪声','原噪声 0.25','小噪声 0.05'];colors=['#2563eb','#ea580c','#059669']
values=[report['summary'][m]['success_rate']*100 for m in modes]
bounds=np.array([report['summary'][m]['success_rate_wilson_95'] for m in modes])*100
axes[0].bar(np.arange(3),values,color=colors,width=.6)
axes[0].errorbar(np.arange(3),values,yerr=np.stack([np.array(values)-bounds[:,0],bounds[:,1]-values]),fmt='none',ecolor='#111827',capsize=5)
for i,m in enumerate(modes):
    s=report['summary'][m];axes[0].text(i,bounds[i,1]+2,f"{values[i]:.1f}%\n{s['successes']}/{s['episodes']}",ha='center',fontproperties=font,fontsize=10)
axes[0].set_xticks(np.arange(3),labels,fontproperties=font);axes[0].set_ylim(0,max(60,float(bounds.max()+12)))
axes[0].set_ylabel('成功率（%）',fontproperties=font);axes[0].set_title('相同来球与初态：成功率及 95% 区间',fontproperties=font)
for outcome,color,marker,label in [('success','#059669','o','成功'),('timeout','#dc2626','s','超时'),('failed','#64748b','x','球掉落 / 离开工作区')]:
    subset=[r for r in records if r['outcome']==outcome]
    axes[1].scatter([r['max_hold_steps']/60 for r in subset],[r['discounted_return'] for r in subset],c=color,marker=marker,s=22,alpha=.55,label=label)
axes[1].axvline(.5,color='#0f172a',linestyle='--',linewidth=1)
axes[1].set_xlabel('回合内最长连续抓握（秒）',fontproperties=font);axes[1].set_ylabel('折扣回报（γ=0.99）',fontproperties=font)
axes[1].set_title('高奖励是否对应连续抓稳',fontproperties=font);axes[1].legend(prop=font,fontsize=9,loc='lower right')
for ax in axes:
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
fig.suptitle('项目09：冻结同一权重，固定课程 0.58，成功判据保持不变',fontproperties=font,fontsize=13)
fig.tight_layout(rect=[0,0,1,.92]);fig.savefig(args.output,bbox_inches='tight');print(str(args.output.resolve()))
