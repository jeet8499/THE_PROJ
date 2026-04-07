"""inference.py — GoldTrading-XAU/USD-v4 — strict [START][STEP][END] format"""
import os,sys,json,argparse
from datetime import datetime,timezone
import numpy as np
import torch
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from env.trading_env import TradingEnv
from env.tasks import TASK_REGISTRY,grade_episode
from agent.state_encoder import StateEncoder
from agent.policy_network import DQNPolicy
from agent.hybrid_agent import LLMBiasExtractor,HYBRID_STATE_DIM

API_BASE_URL=os.environ.get("API_BASE_URL","https://api.openai.com/v1")
MODEL_NAME=os.environ.get("MODEL_NAME","rule-based-oracle")
HF_TOKEN=os.environ.get("HF_TOKEN","")

def load_policy(ckpt=None):
    pol=DQNPolicy(state_dim=HYBRID_STATE_DIM,hidden=128)
    if ckpt and os.path.exists(ckpt):
        pol.load_state_dict(torch.load(ckpt,map_location="cpu")["policy"])
    pol.eval();return pol

def build_action(idx,obs):
    p=obs.get("price",2300);f72=obs.get("fib_72",p-20);f85=obs.get("fib_85",p+20)
    buf={"low":4,"medium":8,"high":15}.get(obs.get("volatility","medium"),8)
    if idx==0:return{"decision":"hold","stop_loss":0.0,"take_profit":0.0}
    if idx==1:sl=f72-buf;return{"decision":"buy","stop_loss":round(sl,2),"take_profit":round(p+2.2*(p-sl),2)}
    sl=f85+buf;return{"decision":"sell","stop_loss":round(sl,2),"take_profit":round(p-2.2*(sl-p),2)}

def run_task(task_id,policy,llm,n_eps=5):
    grader=TASK_REGISTRY[task_id];scores=[];enc=StateEncoder()
    for ep in range(1,n_eps+1):
        env=TradingEnv(difficulty=grader.difficulty,episode_len=grader.episode_len,noise_level=grader.noise_level)
        obs,_=env.reset();enc.reset();done=False
        print(f"\n[START]")
        print(f"task_id: {task_id}")
        print(f"episode: {ep}")
        print(f"timestamp: {datetime.now(timezone.utc).isoformat()}")
        print(f"model: {MODEL_NAME}")
        while not done:
            in_pos=obs.get("position","flat")!="flat"
            # Always go through LLMBiasExtractor.get_bias() so OpenAI client
            # path is used when credentials are provided; deterministic fallback
            # is applied when credentials are not set.
            aug=np.concatenate([enc.encode(obs),llm.get_bias(obs)])
            idx,_=policy.act(aug,epsilon=0.0,in_position=in_pos)
            action=build_action(idx,obs)
            obs,reward,terminated,truncated,info=env.step(action)
            enc.update_from_info(info);done=terminated or truncated
            print(f"[STEP]")
            print(f"step: {info['step']}")
            print(f"observation: {json.dumps({k:obs.get(k) for k in ('price','position','equity','total_pnl','trend')})}")
            print(f"action: {json.dumps({k:action[k] for k in ('decision','stop_loss','take_profit')})}")
            print(f"reward: {reward}")
            print(f"sl_hit: {info.get('sl_hit',False)}")
            print(f"tp_hit: {info.get('tp_hit',False)}")
        summary=env.episode_summary();score=grade_episode(task_id,summary);scores.append(score)
        print(f"[END]")
        print(f"task_id: {task_id}")
        print(f"episode: {ep}")
        print(f"score: {score:.4f}")
        print(f"total_pnl: {summary.get('total_pnl',0):+.2f}")
        print(f"n_trades: {summary.get('n_trades',0)}")
        print(f"win_rate: {summary.get('win_rate',0):.1f}%")
        print(f"max_drawdown: {summary.get('max_drawdown',0):.3f}%")
        print("-"*60)
    avg=round(float(np.mean(scores)),4)
    return{"task_id":task_id,"avg_score":avg,"scores":scores,"difficulty":grader.difficulty}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task",default="all",choices=["all"]+list(TASK_REGISTRY.keys()))
    p.add_argument("--episodes",type=int,default=5)
    p.add_argument("--checkpoint",type=str,default="checkpoints/hybrid_dqn_ep600.pt")
    p.add_argument("--output",type=str,default=None)
    args=p.parse_args();policy=load_policy(args.checkpoint);llm=LLMBiasExtractor()
    task_ids=list(TASK_REGISTRY.keys()) if args.task=="all" else [args.task]
    results=[run_task(tid,policy,llm,args.episodes) for tid in task_ids]
    print("\n"+"="*60);print("  BASELINE SCORES — GoldTrading-XAU/USD-v4");print("="*60)
    for r in results:print(f"  {r['task_id']:<20} {r['difficulty']:<12} {r['avg_score']:>8.4f}")
    print(f"\n  Overall: {round(float(np.mean([r['avg_score'] for r in results])),4):.4f}")
    if args.output:
        os.makedirs(os.path.dirname(args.output) or".",exist_ok=True)
        with open(args.output,"w") as f:json.dump(results,f,indent=2)
        print(f"  Saved → {args.output}")
if __name__=="__main__":main()
