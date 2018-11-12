--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "options.train"

if not dqn then
    require "initenv"
end

local opt = globalDQNOptions

--- General setup.
local game_env, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

-- Take one single initial step to get kicked-off...
local screen, reward, terminal = game_env:getState()

local last_step_log_time = sys.clock()
local win = nil
while step < opt.steps do
    step = step + 1 
    local action = agent:perceive(reward, screen, terminal)

    -- game over? get next game!
    if not terminal then
    
        -- Play the selected action in the emulator. 
        -- Record the resulting screen, reward, and whether this was terminal.
        screen, reward, terminal = game_env:step(action, true)
            
      -- Spam the console.
      if opt.verbose > 3 and reward ~= 0 then
        print("Reward: " .. reward)
      end
    else
      if opt.random_starts > 0 then
          screen, reward, terminal = game_env:nextRandomGame()
            
          -- Spam the console.
          if opt.verbose > 3 then
            print("New random episode.")
          end
      else
          screen, reward, terminal = game_env:newGame()
            
          -- Spam the console.
          if opt.verbose > 3 then
            print("New episode.")
          end
       end
    end

    -- Logging...
    if step % 10000 == 0 then
       local elapsed_step_time = sys.clock() - last_step_log_time
       last_step_log_time = sys.clock()
       print("Steps: " .. step .. " Time: " .. elapsed_step_time)
    end

    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        print("Epsilon: ", agent.ep)
        agent:report()
        
        -- Save the hist_len most recent frames.
        if opt.verbose > 3 then
          agent:printRecent()
        end
        collectgarbage()
    end

    if step%1000 == 0 then collectgarbage() end

    if step % opt.eval_freq == 0 and step > learn_start then
    
        print("***********")
        print("Starting evaluation!")
        print("***********")

        screen, reward, terminal = game_env:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action = agent:perceive(reward, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = game_env:step(action)

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward
            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if opt.verbose > 3 and reward ~= 0 then
              print("Episode Reward: " .. episode_reward)
              print ("Number of Rewards: " .. nrewards)
            end

            if terminal then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1

                if opt.verbose > 3 then
                  print("Total Reward: " .. total_reward)
                end
                if opt.random_starts > 0 then
                  screen, reward, terminal = game_env:nextRandomGame()
                else
                  screen, reward, terminal = game_env:newGame()
                end
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = '../networks/' .. filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print("***********")    
        print('Saved:', filename .. '.t7')
        print("***********")
        io.flush()
        collectgarbage()
    end
end
