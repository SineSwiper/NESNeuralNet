-- Code for human training loading/filling
local Env = torch.getmetatable('NES.NesEnv')

function Env:fillTrainingCache()
    local dir = table.concat({ROOT_PATH, 'movies', 'training', self.config.gamename}, "/")

    -- Find movie files to use and sort them
    local fm_files = {}
    for file in lfs.dir(dir) do
        local full_path = dir .. '/' .. file
        if file:match('.fm2$') and lfs.attributes(full_path, 'mode') == 'file' then
            table.insert(fm_files, full_path)
        end
    end
    table.sort(fm_files)

    -- Read the movie files
    for _, full_path in ipairs(fm_files) do
        local file = full_path:match("^.+/(.+)$")
        local skip_frames = self.romEnv:getNumSkipFrames()

        print("Loading " .. file .. " into training cache")
        for line in io.lines(full_path) do
            --- Header sanity checks
            if line:match('^%w+ .+$') then
                local key, val = line:match('^(%w+) (.+)$')

                if     key == 'version' and val ~= '3' then
                    print('    WARNING: Only a V' .. val .. ' (not V3) FCEUX movie file')
                elseif key == 'romFilename' then
                    -- Only do a loose check here, since filename flags may vary.
                    -- Adjust as necessary...
                    local mgamename = val:match("^[%w _-&'%.!]+"):gsub("[ '%.!]", '')
                    if not mgamename:match(self.config.gamename) then
                        print('    WARNING: ROM filenames are different: ' .. mgamename .. ' vs. ' .. self.config.gamename)
                    end
                elseif key == 'romChecksum' and val ~= rom.gethash('base64') then
                    print('    WARNING: Movie checksum does not match loaded ROM')
                elseif key == 'palFlag' and val ~= '0' then
                    print('    WARNING: PAL flag enabled')

                -- TODO: fourscore, port0/1/2 checks

                elseif key == 'FDS' and val ~= '0' then
                    print('    WARNING: Famicom Disk System movie file; may not work right...')
                elseif key == 'NewPPU' and val ~= '0' then
                    print('    WARNING: Moving using New PPU core; may not work right...')
                end

            -- Frame line
            elseif line:match('^|%d+|.*|.*|.*|.*$') then
                if skip_frames > 0 then
                    skip_frames = skip_frames - 1
                    -- Insert reset line
                    if skip_frames == 0 then table.insert(self.trainingCache, '|1||||') end
                else
                    -- Frame data; process this later
                    table.insert(self.trainingCache, line)
                end
            end
        end
    end
end

function Env:fillTrainingActions(actions, step, min, len)
    if table.getn(self.trainingCache) == 0 then return end

    -- Random bits
    if not len then len = torch.random(1, 300) end  -- upwards of 5 seconds
    if not min then min = torch.random(table.getn(self.trainingCache) - len) end
    local max = min + len - 1

    -- Button table prep
    local btnSet = self.romEnv:getLegalButtonSet()
    local idxSet = {}  -- reverse lookup
    for _, btn in pairs(btnSet) do
        local key = btn[1] .. ',' .. NES.BUTTON2FRAME[btn[2]]
        idxSet[key] = _
    end

    -- Process frame lines (with dynamic loop control)
    local i = min
    while i <= max do
        --- XXX: Only supports Player 1 right now
        local movie_line = self.trainingCache[i]
        local flags, player_one = movie_line:match('^|(%d+)|([^|]*)|')

        -- Special action to reset the game.  Only used if it appears that
        -- full human training is requested.  Note that in normal *.fm2 files
        -- the rest of the line applies (but is usually blank), but not in the
        -- manner we're using it above (with "|1||||").
        if flags == '1' then
            if step == 1 and min == 1 then table.insert(actions, { 'reset' }) end
            i = i + 1

        -- Early stages of learning needs more active development, so skip
        -- blank actions when the step isn't very accurate, anyway.
        elseif (step >= 4 and not player_one:match('[A-Z]')) then
            i = i + 1
            max = math.min(max + 1, table.getn(self.trainingCache))

        -- Standard action
        else
            local action = {}
            for j=1,player_one:len(),1 do
                local b = player_one:sub(j, j)
                if b ~= '.' and NES.FRAME2BUTTON[b] then
                    local key = '1,' .. b
                    local val = idxSet[key]
                    if val then table.insert(action, val) end
                end
            end

            table.insert(actions, action)
            i = i + step
        end
    end
end
