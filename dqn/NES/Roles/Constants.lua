-- Standard NES specs
if not NES then NES = {} end

NES.RAM_LENGTH    = 2048
NES.SCREEN_WIDTH  = 256
NES.SCREEN_HEIGHT = 240

NES.FRAME2BUTTON = {
    ['R']='right',
    ['L']='left',
    ['D']='down',
    ['U']='up',
    ['T']='start',
    ['S']='select',
    ['B']='B',
    ['A']='A',
}
NES.BUTTON2FRAME = {}
NES.BUTTON_SET   = {}

for i = 1, 8 do
    local letter = string.sub('UDLRBAST', i, i)
    local button = NES.FRAME2BUTTON[letter]
    table.insert(NES.BUTTON_SET, button)
    NES.BUTTON2FRAME[button] = letter
end
