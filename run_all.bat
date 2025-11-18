@echo off
for /l %%i in (1,1,13) do (
    echo Running for deer/%%i.mp4
    python deer.py --use_pytorch --pytorch_model ./deer.pt --unsafe_dist 100 --danger_dist 100 --video ./deer/%%i.mp4
)