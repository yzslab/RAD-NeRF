import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()

base_dir = os.path.dirname(args.video)

with open(os.path.join(base_dir, "timestamps.txt"), "r") as f:
    with open(os.path.join(base_dir, "clips.txt"), "w") as c:
        counter = 0
        while True:
            # read line by line from timestamps.txt
            timestamp = f.readline()
            if not timestamp:
                break
            timestamp = timestamp.strip()
            if timestamp == "":
                continue
            timestamp = timestamp.split("-")

            # retrieve start time and stop time
            start = timestamp[0]
            stop = timestamp[1]
            # append "00:" prefix to keep the format match to hh:mm:ss
            if start.count(":") == 1:
                start = "00:" + start
            if stop.count(":") == 1:
                stop = "00:" + stop

            # build output path
            output_path = os.path.join(base_dir, "clip-{}.mp4".format(counter))

            # create child process
            pid = os.fork()
            if pid < 0:
                exit(1)

            if pid > 0:
                # parent process
                # append clip filename to txt file
                c.write("file '{}'\n".format(os.path.basename(output_path)))
                counter += 1
            else:
                # run ffmpeg on child process
                os.execvp("ffmpeg", [
                    "ffmpeg",
                    "-i",
                    args.video,
                    "-ss",
                    start,
                    "-to",
                    stop,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    "-y",
                    output_path
                ])
                exit(1)

# wait all ffmpeg finish processing video
while True:
    try:
        pid, status = os.wait()
        if status != 0:
            exit(1)
    except ChildProcessError:
        break

# merge clips
os.chdir(base_dir)
os.system("ffmpeg " + " ".join([
    "-f",
    "concat",
    "-i",
    "clips.txt",
    "-c",
    "copy",
    "-y",
    "clips-merged-raw.mp4",
]))
os.system("ffmpeg -i clips-merged-raw.mp4 -filter:v fps=fps=25 -y clips-merged-25fps.mp4")
os.execvp("stty", ["stty", "sane"])
