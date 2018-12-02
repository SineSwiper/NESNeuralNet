-- Useful for tracking overall load time on step 1
require "sys"
local last_step_log_time = sys.clock()

require "options.train"
--require "options.test"

if not dqn then
    require "initenv"
end

local opt = globalDQNOptions

-- TODO: Change other code to use ROOT_PATH more
opt.network = table.concat({ROOT_PATH, 'networks', opt.name .. '.t7'}, "/")

-- General setup.
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

local heatmap

-- Used for %X trickery
local time_offset     = os.date('%z', 0)
local time_offset_sec =
    math.floor(time_offset / 100) * 3600 +                               -- hour
    (time_offset:sub(-2) * 60 * tonumber(time_offset:sub(1, 1) .. "1"))  -- min * sign

-- Take one single initial step to get kicked-off...
local screen, reward, terminal = game_env:getState()

-- Start with full-on human training data
local in_human_training = opt.human_training

if in_human_training then
    game_env._actrep = 1
    agent.training_actions = game_env:fillAllTrainingActions()
end

local human_training_actions = table.getn(agent.training_actions)

-- Don't enable graphics if we don't have to
if not opt.env_params.useRGB then
    emu.setrenderplanes(false, false)
end

-- Main loop
while step < opt.steps do
    -- Human training
    if in_human_training then
        if table.getn(agent.training_actions) > 0 then
            -- New training set; reset game
            if agent.training_actions[1][1] == 'reset' then
                print("\nNew human training movie")
                print("========================")
                game_env.NesEnv:resetGame()
                table.remove(agent.training_actions, 1)
            end
        else
            -- End of human training
            in_human_training = false
            game_env.NesEnv:resetGame()
            game_env._actrep = opt.actrep
        end
    elseif table.getn(agent.training_actions) == 0 and torch.uniform() < 0.10 then
        agent.training_actions = game_env:fillTrainingActions()
    end

    step = step + 1
    local action
    if in_human_training then
        action = agent:perceive(reward, screen, terminal, false, 2)  -- 2 = human training
    else
        action = agent:perceive(reward, screen, terminal, opt.testing, opt.testing_ep)
    end

    -- game over? get next game!
    if not terminal then

        -- Play the selected action in the emulator.
        -- Record the resulting screen, reward, and whether this was terminal.
        screen, reward, terminal = game_env:step(action)

        -- Spam the console.
        if opt.verbose > 3 and reward ~= 0 then
            print("Reward: " .. reward)
        end
    else
        -- Should we record an example movie?
        if torch.uniform() < opt.example_mp and not in_human_training then
            local basename = opt.name .. '_s' .. step .. '.fm2'
            local filename = table.concat({ROOT_PATH, 'movies', 'examples', basename}, "/")
            game_env:shouldRecordMovie(filename)
        end

        screen, reward, terminal = game_env:newGame()

        -- Spam the console.
        if opt.verbose > 2 then
            print("New episode")
        end
    end

    -- Show heatmap
    if opt.heatmap then
        if step % agent.update_freq == 0 and agent.grad_input then
            heatmap = agent:get_grad_input()
        end
        if heatmap then game_env.NesEnv:displayHeatmap(heatmap) end
    end

    -- Logging...
    if step == 1 or step % 10000 == 0 then
        local elapsed_step_time = os.difftime(sys.clock(), last_step_log_time)
        last_step_log_time = sys.clock()
        local elapsed_step_time_str = os.date("%X", elapsed_step_time + 86400 - time_offset_sec)

        local log_str = "Steps: " .. step .. ", Time: " .. elapsed_step_time_str

        if opt.gpu and opt.gpu >= 0 then
            local freeVRAM, totalVRAM = cutorch.getMemoryUsage(opt.gpu)
            local usedVRAM   = totalVRAM - freeVRAM
            local perVRAM    = math.floor(usedVRAM / totalVRAM * 1000 + 0.5) / 10
            local usedVRAMGB = math.floor(usedVRAM / 1024^3 * 10 + 0.5) / 10

            log_str = log_str .. ", VRAM: " .. usedVRAMGB .. "GiB (" .. perVRAM .. "%)"
        end
        print(log_str)
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

    if not in_human_training and step % opt.eval_freq == 0 and step > learn_start then

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
                screen, reward, terminal = game_env:newGame()
            end
        end

        eval_time  = os.difftime(sys.clock(), eval_time)
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        -- #reward_history ? torch.Tensor(...) : 0, but this is Lua...
        local reward_max = 0
        if #reward_history > 0 then reward_max = torch.Tensor(reward_history):max() end

        if total_reward > reward_max then
            local total_reward_int = math.floor(total_reward + 0.5)
            if opt.verbose > 2 then
                print( "Found a better network: " .. total_reward_int .. " vs. " .. math.floor(reward_max + 0.5) )
            end

            -- Be very careful to not clone a large CUDA object
            agent.best_network = agent.network:float():clone()
            collectgarbage()
            if opt.gpu and opt.gpu >= 0 then
                agent.network:cuda()
            end

            -- If it's that good, then record it!
            if not in_human_training then
                local basename = opt.name .. '_r' .. total_reward_int .. '.fm2'
                local filename = table.concat({ROOT_PATH, 'movies', 'best', basename}, "/")
                game_env:shouldRecordMovie(filename)
            end
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

        local training_rate = game_env._actrep*opt.eval_freq / time_dif

        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local filename = opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end

        filename = table.concat({ROOT_PATH, 'networks', filename}, "/")
        torch.save(filename .. '.t7', {
            model          = agent.network,
            best_model     = agent.best_network,
            reward_history = reward_history,
            reward_counts  = reward_counts,
            episode_counts = episode_counts,
            time_history   = time_history,
            v_history      = v_history,
            td_history     = td_history,
            qmax_history   = qmax_history,
            arguments      = opt
        })
        if opt.saveNetworkParams then
            local nets = {network=agent.w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end

        print("***********")
        print('Saved:', filename .. '.t7')
        print("***********")
        io.flush()
        collectgarbage()
    end
end
