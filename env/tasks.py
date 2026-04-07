"""env/tasks.py — Three distinct TaskGrader classes"""
from typing import Any,Dict
class EasyGrader:
    task_id="task_easy";task_name="Direction-only signal detection"
    difficulty="easy";noise_level=0.10;episode_len=20
    def grade(self,summary:Dict[str,Any])->float:
        steps_log=summary.get("steps_log",[])
        if not steps_log:return 0.0
        correct=0;total=0;regime=summary.get("regime","range")
        for s in steps_log:
            dec=s["action"].get("decision","hold");total+=1
            if dec=="hold" and regime=="range":correct+=1
            elif dec=="buy" and regime=="bullish":correct+=1
            elif dec=="sell" and regime=="bearish":correct+=1
        return round(min(1.0,max(0.0,correct/total)) if total>0 else 0.0,4)
    def description(self):return "Predict direction aligned with market regime. Score=accuracy."
class MediumGrader:
    task_id="task_medium";task_name="Full trade: direction + SL placement"
    difficulty="medium";noise_level=0.30;episode_len=30
    def grade(self,summary:Dict[str,Any])->float:
        steps_log=summary.get("steps_log",[]);regime=summary.get("regime","range")
        trades=summary.get("trades",[])
        if not steps_log:return 0.0
        dc=0;dt=0;ss=0.0;sn=0;tp_hit=any(t.get("exit_reason")=="tp_hit" for t in trades)
        for s in steps_log:
            dec=s["action"].get("decision","hold")
            if dec=="hold":continue
            dt+=1
            if(dec=="buy" and regime=="bullish")or(dec=="sell" and regime=="bearish"):dc+=1
            sl=s["action"].get("stop_loss",0)
            if sl>0:
                risk=abs(s["info"].get("price",0)-sl)
                ss+=(1.0 if 5<=risk<=50 else(0.2 if risk<5 else 0.5));sn+=1
        d=dc/dt if dt>0 else 0.0;s2=ss/sn if sn>0 else 0.0;t=1.0 if tp_hit else 0.0
        return round(min(1.0,max(0.0,0.50*d+0.30*s2+0.20*t)),4)
    def description(self):return "Direction+SL quality+TP hit. Score=50/30/20."
class HardGrader:
    task_id="task_hard";task_name="Multi-trade with drawdown constraint DD<2%"
    difficulty="hard";noise_level=0.55;episode_len=40;DD_LIMIT=2.0;MIN_TRADES=2
    def grade(self,summary:Dict[str,Any])->float:
        pnl=summary.get("pnl_pct",0.0);dd=summary.get("max_drawdown",100.0)
        wr=summary.get("win_rate",0.0)/100.0;nt=summary.get("n_trades",0)
        ps=min(1.0,max(0.0,(pnl+2.0)/4.0))
        ds=(1.0 if dd<=self.DD_LIMIT else max(0.0,1.0-(dd-self.DD_LIMIT)/self.DD_LIMIT) if dd<=self.DD_LIMIT*2 else 0.0)
        ws=min(1.0,max(0.0,(wr-0.40)/0.60)) if nt>0 else 0.0
        fs=min(1.0,nt/self.MIN_TRADES)
        return round(min(1.0,max(0.0,0.35*ps+0.25*ds+0.25*ws+0.15*fs)),4)
    def description(self):return "Profitable multi-trade, max drawdown<2%."
TASK_REGISTRY={"task_easy":EasyGrader(),"task_medium":MediumGrader(),"task_hard":HardGrader()}
def grade_episode(task_id:str,summary:dict)->float:
    if task_id not in TASK_REGISTRY:raise ValueError(f"Unknown task: {task_id}")
    return TASK_REGISTRY[task_id].grade(summary)
def list_tasks():
    return[{"task_id":t.task_id,"name":t.task_name,"difficulty":t.difficulty}for t in TASK_REGISTRY.values()]
