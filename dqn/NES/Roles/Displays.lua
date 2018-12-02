-- Code for displaying extra info on the emulator screen
local Env = torch.getmetatable('NES.NesEnv')

function Env:displayButtons(btns)
    local xOff, yOff = self.romEnv:btnDisplayPos()

    for player, pBtns in pairs(btns) do
        for i, btn in pairs(NES.BUTTON_SET) do
            local letter = NES.BUTTON2FRAME[btn]
            local color = 'white'
            if pBtns[btn] == true then color = 'red' end

            local x = i * 7 + xOff
            local y = (player-1) * 10 + yOff

            gui.text(x, y, letter, color, 'black')
        end
    end
end

function Env:displayHeatmap(heatmap, level)
    local xOff, yOff = self.romEnv:heatmapDisplayPos()

    -- Compress it down, if required
    if heatmap:nDimension() == 4 then
        -- 4D -> 2D (MBSize x Frames x X x Y --> X x Y)

        -- Try upsampling first
        if heatmap:size(3) < 84 and heatmap:size(4) < 84 then
            local upsampler = nn.UpSampling({oheight=84, owidth=84}, 'linear')
            if torch.typename(heatmap):match('Cuda') then upsampler:cuda() end
            heatmap = upsampler:forward(heatmap)
        end

        local avg = heatmap:mean(2)
        avg = avg:mean(1)
        avg = avg:squeeze()
        heatmap = avg
    elseif heatmap:nDimension() == 3 then
        -- 3D -> 2D (Frames x X x Y --> X x Y)
        local avg = heatmap:mean(1)
        avg = avg:squeeze()
        heatmap = avg
    end

    local maxY = heatmap:size(1)
    local maxX = heatmap:size(2)
    local maxQ = math.max( heatmap:min() * -1, heatmap:max() )

    -- Display level
    gui.text(xOff - 7, yOff, level, 'white', 'black')

    for y=1,maxY do
        -- Use a form of RLE compression to optimize the drawing
        local prevColor, prevX

        for x=1,maxX do
            local q = heatmap[y][x]
            local r,g,b,a = 0,0,0,255
            if     q > 0 then
                g = math.floor( q / maxQ * 255, 255)
            elseif q < 0 then
                r = math.floor(-q / maxQ * 255, 255)
            end

            -- faster to pass this than have FCEUX calculate a Lua table
            local c = r * 16777216 + g * 65536 + r * 256 + a

            -- Figure out if we have to draw or not
            if not prevColor then
                prevColor, prevX = c, x
            elseif prevColor == c then
                -- Do nothing; wait until it changes
            else
                -- Different color; we're drawing something...
                if prevX == x - 1 then
                    gui.pixel(prevX + xOff, y + yOff, prevColor)
                else
                    gui.line(prevX + xOff, y + yOff, x - 1 + xOff, y + yOff, prevColor)
                end

                prevColor, prevX = c, x
            end
        end

        -- Draw the final line/pixel for this row
        if prevX == maxX then
            gui.pixel(prevX + xOff, y + yOff, prevColor)
        else
            gui.line(prevX + xOff, y + yOff, maxX + xOff, y + yOff, prevColor)
        end
    end
end
