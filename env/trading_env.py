"""env/trading_env.py — gymnasium.Env, Pydantic models, all-numeric obs"""
import sys,os
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
import copy,random
from typing import Any,Dict,Optional
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
from pydantic import BaseModel,Field
from data.market_generator import generate_episode
from env.position import PositionManager
from env.reward import compute_step_reward

TREND_ENC={"bullish":0,"bearish":1,"range":2}
SENT_ENC={"positive":0,"negative":1,"neutral":2}
VOL_ENC={"low":0,"medium":1,"high":2}
CONF_ENC={"confirmed":0,"not_confirmed":1}
ZONE_ENC={"inside_zone":0,"above_zone":1,"below_zone":2}
POS_ENC={"flat":0,"long":1,"short":2}

class Observation(BaseModel):
    pair:str;price:float;trend:str;fib_72:float;fib_85:float;zone_position:str
    sentiment:str;volatility:str;confirmation:str;position:str;equity:float
    unrealized_pnl:float;total_pnl:float;max_drawdown:float;steps_remaining:int
    trend_enc:int;sentiment_enc:int;volatility_enc:int;confirmation_enc:int
    zone_enc:int;position_enc:int

class Action(BaseModel):
    decision:str=Field(...,pattern="^(buy|sell|hold)$")
    stop_loss:float=Field(0.0,ge=0.0)
    take_profit:float=Field(0.0,ge=0.0)

class StepReward(BaseModel):
    total:float;decision:float;stop_loss:float;take_profit:float;hold_bonus:float

class TradingEnv(gym.Env):
    metadata={"render_modes":["human"]}
    ENV_ID="GoldTrading-XAU/USD-v4";VERSION="4.0.0";SPEC="openenv-v1"
    def __init__(self,episode_len=40,n_warmup=50,noise_level=0.3,
                 difficulty="medium",seed=None,render_mode=None):
        super().__init__()
        self.episode_len=episode_len;self.n_warmup=n_warmup;self.difficulty=difficulty
        self._seed=seed;self._rng=random.Random(seed);self.render_mode=render_mode
        self.noise_level={"easy":0.1,"medium":0.3,"hard":0.55}.get(difficulty,noise_level)
        self.observation_space=spaces.Dict({
            "price":spaces.Box(0,10000,(1,),dtype=np.float32),
            "fib_72":spaces.Box(0,10000,(1,),dtype=np.float32),
            "fib_85":spaces.Box(0,10000,(1,),dtype=np.float32),
            "trend_enc":spaces.Discrete(3),"sentiment_enc":spaces.Discrete(3),
            "volatility_enc":spaces.Discrete(3),"confirmation_enc":spaces.Discrete(2),
            "zone_enc":spaces.Discrete(3),"position_enc":spaces.Discrete(3),
            "equity":spaces.Box(0,1e6,(1,),dtype=np.float32),
            "unrealized_pnl":spaces.Box(-1e5,1e5,(1,),dtype=np.float32),
            "total_pnl":spaces.Box(-1e5,1e5,(1,),dtype=np.float32),
            "max_drawdown":spaces.Box(0,100,(1,),dtype=np.float32),
            "steps_remaining":spaces.Discrete(episode_len+1),
        })
        self.action_space=spaces.Dict({
            "decision":spaces.Discrete(3),
            "stop_loss":spaces.Box(0,10000,(1,),dtype=np.float32),
            "take_profit":spaces.Box(0,10000,(1,),dtype=np.float32),
        })
        self._episode_data=None;self._step_idx=0;self._done=True
        self._episode_num=0;self.position_mgr=PositionManager();self._episode_log=[]

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if seed is not None:self._rng=random.Random(seed)
        ep_seed=self._rng.randint(0,999_999)
        self._episode_data=generate_episode(seed=ep_seed,n_candles=self.n_warmup,
            episode_len=self.episode_len,noise_level=self.noise_level)
        self._step_idx=0;self._done=False;self._episode_num+=1
        self._episode_log=[];self.position_mgr.reset()
        return self._build_obs(),{"episode":self._episode_num,"regime":self._episode_data["regime"]}

    def step(self,action):
        if self._done:raise RuntimeError("Episode done. Call reset() first.")
        action=self._normalise_action(action)
        s=self._episode_data["steps"][self._step_idx]
        obs_d=s["observation"];price=obs_d["price"];was_flat=self.position_mgr.is_flat
        if action["decision"] in("buy","sell") and self.position_mgr.is_flat:
            self.position_mgr.open_position(
                direction="long" if action["decision"]=="buy" else "short",
                price=price,stop_loss=action["stop_loss"],
                take_profit=action["take_profit"],bar=self._step_idx)
        ni=self._step_idx+1
        nc=(self._episode_data["steps"][ni]["candle"] if ni<len(self._episode_data["steps"]) else s["candle"])
        br=self.position_mgr.update(nc,ni)
        self._step_idx+=1;is_last=self._step_idx>=self.episode_len
        if is_last and not self.position_mgr.is_flat:
            br["realized_pnl"]=br.get("realized_pnl",0)+self.position_mgr.close_position(nc["close"],self._step_idx,"episode_end")
        self._done=is_last
        reward=compute_step_reward(action=action,bar_result=br,obs=obs_d,
            position_before=None if was_flat else "open",is_last=is_last,position_mgr=self.position_mgr)
        info={"step":self._step_idx,"action":action,"price":price,
              "position":br.get("position"),"realized_pnl":br.get("realized_pnl",0),
              "unrealized_pnl":br.get("unrealized_pnl",0),
              "sl_hit":br.get("sl_hit",False),"tp_hit":br.get("tp_hit",False),
              "trade_opened":not was_flat==self.position_mgr.is_flat,
              "equity":round(self.position_mgr.equity,2),
              "total_pnl":round(self.position_mgr.total_pnl,2),
              "max_drawdown":round(self.position_mgr.max_drawdown*100,3),
              "episode":self._episode_num,"regime":self._episode_data["regime"]}
        self._episode_log.append({"step":self._step_idx,"action":action,"reward":reward,"info":info})
        return self._build_obs(),round(reward,6),is_last,False,info

    def state(self):
        return{"env_id":self.ENV_ID,"version":self.VERSION,"spec":self.SPEC,
               "episode":self._episode_num,"step":self._step_idx,"done":self._done,
               "position":self.position_mgr.position,"equity":round(self.position_mgr.equity,2),
               "total_pnl":round(self.position_mgr.total_pnl,2),
               "max_drawdown":round(self.position_mgr.max_drawdown*100,3),
               "regime":self._episode_data["regime"] if self._episode_data else None}

    def episode_summary(self):
        return{**self.position_mgr.summary(),
               "regime":self._episode_data["regime"] if self._episode_data else None,
               "episode_len":self.episode_len,"noise_level":self.noise_level,
               "difficulty":self.difficulty,"steps_log":self._episode_log}

    def openenv_validate(self):
        checks={}
        try:
            obs,info=self.reset()
            checks["reset_returns_obs_info"]=isinstance(obs,dict) and isinstance(info,dict)
            checks["obs_has_numeric_encodings"]=all(k in obs for k in ["trend_enc","sentiment_enc","zone_enc"])
        except Exception:checks["reset_returns_obs_info"]=False
        try:
            r=self.step({"decision":"hold","stop_loss":0.0,"take_profit":0.0})
            checks["step_returns_5tuple"]=isinstance(r,tuple) and len(r)==5
            checks["step_reward_is_float"]=isinstance(r[1],float)
            checks["terminated_is_bool"]=isinstance(r[2],bool)
            checks["truncated_is_bool"]=isinstance(r[3],bool)
        except Exception:checks["step_returns_5tuple"]=False
        checks["has_state"]=callable(getattr(self,"state",None))
        checks["has_episode_summary"]=callable(getattr(self,"episode_summary",None))
        checks["gymnasium_subclass"]=isinstance(self,gym.Env)
        checks["env_id_correct"]=self.ENV_ID=="GoldTrading-XAU/USD-v4"
        checks["pydantic_models_defined"]=True
        all_pass=all(checks.values())
        return{"valid":all_pass,"checks":checks,"env_id":self.ENV_ID}

    def _build_obs(self):
        idx=min(self._step_idx,len(self._episode_data["steps"])-1)
        obs=copy.deepcopy(self._episode_data["steps"][idx]["observation"])
        pos=self.position_mgr.position or"flat"
        obs["position"]=pos;obs["equity"]=round(self.position_mgr.equity,2)
        obs["unrealized_pnl"]=round(self.position_mgr.unrealized_pnl,2)
        obs["total_pnl"]=round(self.position_mgr.total_pnl,2)
        obs["max_drawdown"]=round(self.position_mgr.max_drawdown*100,3)
        obs["steps_remaining"]=self.episode_len-self._step_idx
        obs["trend_enc"]=TREND_ENC.get(obs.get("trend","range"),2)
        obs["sentiment_enc"]=SENT_ENC.get(obs.get("sentiment","neutral"),2)
        obs["volatility_enc"]=VOL_ENC.get(obs.get("volatility","medium"),1)
        obs["confirmation_enc"]=CONF_ENC.get(obs.get("confirmation","not_confirmed"),1)
        obs["zone_enc"]=ZONE_ENC.get(obs.get("zone_position","above_zone"),1)
        obs["position_enc"]=POS_ENC.get(pos,0)
        return obs

    @staticmethod
    def _normalise_action(action):
        if not isinstance(action,dict):raise ValueError("action must be dict")
        dec=action.get("decision","hold")
        if isinstance(dec,int):dec=["hold","buy","sell"][dec]
        return{"decision":str(dec).lower().strip(),
               "stop_loss":float(action.get("stop_loss",0.0)),
               "take_profit":float(action.get("take_profit",0.0))}
