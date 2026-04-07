"""app.py — HF Space Gradio (port 7860). gradio included in requirements.txt."""
import os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import gradio as gr
import numpy as np,torch
from env.trading_env import TradingEnv
from env.tasks import TASK_REGISTRY,grade_episode
from agent.state_encoder import StateEncoder
from agent.policy_network import DQNPolicy
from agent.hybrid_agent import LLMBiasExtractor,HYBRID_STATE_DIM
POLICY=None;LLM=LLMBiasExtractor()
def load_policy():
    global POLICY
    if POLICY:return POLICY
    pol=DQNPolicy(state_dim=HYBRID_STATE_DIM,hidden=128)
    for p in["checkpoints/hybrid_dqn_ep600.pt","checkpoints/dqn_ep500.pt"]:
        if os.path.exists(p):pol.load_state_dict(torch.load(p,map_location="cpu")["policy"]);break
    pol.eval();POLICY=pol;return pol
def run_ep(task_id,seed):
    pol=load_policy();g=TASK_REGISTRY[task_id]
    env=TradingEnv(difficulty=g.difficulty,episode_len=g.episode_len,noise_level=g.noise_level)
    obs,_=env.reset(seed=int(seed));enc=StateEncoder();done=False
    lines=[f"Task: {g.task_name}","─"*55]
    while not done:
        in_pos=obs.get("position","flat")!="flat"
        aug=np.concatenate([enc.encode(obs),LLM._rule_based_bias(obs)])
        idx,_=pol.act(aug,epsilon=0.0,in_position=in_pos)
        p=obs.get("price",0);f72=obs.get("fib_72",p-20);f85=obs.get("fib_85",p+20)
        buf={"low":4,"medium":8,"high":15}.get(obs.get("volatility","medium"),8)
        if idx==0:action={"decision":"hold","stop_loss":0.0,"take_profit":0.0}
        elif idx==1:sl=f72-buf;action={"decision":"buy","stop_loss":round(sl,2),"take_profit":round(p+2.2*(p-sl),2)}
        else:sl=f85+buf;action={"decision":"sell","stop_loss":round(sl,2),"take_profit":round(p-2.2*(sl-p),2)}
        obs,reward,terminated,truncated,info=env.step(action);enc.update_from_info(info);done=terminated or truncated
        out="SL" if info["sl_hit"] else("TP" if info["tp_hit"] else"  ")
        lines.append(f"Step {info['step']:>2}|{action['decision'].upper():4}|${info['price']:>8.2f}|PnL{info['total_pnl']:>+8.2f}|{out}")
    s=env.episode_summary();score=grade_episode(task_id,s)
    lines+=["─"*55,f"Score:{score:.4f} PnL:${s['total_pnl']:+.2f} Trades:{s['n_trades']} WR:{s['win_rate']:.1f}% DD:{s['max_drawdown']:.2f}%"]
    return"\n".join(lines)
def validate():
    v=TradingEnv().openenv_validate()
    return"\n".join([f"{'✓' if ok else '✗'}  {k}" for k,ok in v["checks"].items()]+[f"\nResult:{'PASS' if v['valid'] else 'FAIL'}"])
def tasks_info():return"\n".join(f"{t.task_id} [{t.difficulty}]\n  {t.description()}" for t in TASK_REGISTRY.values())
with gr.Blocks(title="GoldTrading-XAU/USD-v4") as demo:
    gr.Markdown("# 🥇 GoldTrading-XAU/USD-v4 — Hybrid RL + LLM\nOpenEnv · Gymnasium · 3 graded tasks")
    with gr.Tab("Run Episode"):
        with gr.Row():td=gr.Dropdown(list(TASK_REGISTRY.keys()),value="task_easy",label="Task");sn=gr.Number(value=42,label="Seed",precision=0)
        gr.Button("Run",variant="primary").click(run_ep,[td,sn],gr.Textbox(label="Log",lines=28))
    with gr.Tab("Validate"):gr.Button("openenv_validate()").click(validate,outputs=gr.Textbox(label="Result",lines=15))
    with gr.Tab("Tasks"):gr.Button("List tasks").click(tasks_info,outputs=gr.Textbox(label="Tasks",lines=10))
if __name__=="__main__":demo.launch(server_name="0.0.0.0",server_port=7860)
