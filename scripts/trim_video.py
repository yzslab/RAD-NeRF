import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("video")
args = parser.parse_args()

base_dir = os.path.dirname(args.video)

counter = 0
with open(os.path.join(base_dir, "timestamps.txt"), "r") as f:
    with open(os.path.join(base_dir, "clips.txt"), "w") as c:
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
            assert pid >= 0, "error occurred when calling os.fork()"

            if pid > 0:
                # parent process
                # append clip filename to txt file
                c.write("file '{}'\n".format(os.path.basename(output_path)))
                counter += 1
            else:
                # child process
                # redirect stdout and stderr to file
                logfile_fd = os.open(
                    os.path.join(base_dir, os.path.basename(output_path) + ".log"),
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                )
                os.dup2(logfile_fd, 1)
                os.dup2(logfile_fd, 2)
                # close stdin
                os.close(0)
                # run ffmpeg on child process
                os.execvp("ffmpeg", [
                    "ffmpeg",
                    "-nostdin",
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
                assert False, "error occurred when calling os.execvp()"

# wait all ffmpeg finish processing video
with tqdm(range(counter)) as t:
    for i in t:
        try:
            pid, status = os.wait()
            assert status == 0, "error occurred when creating clip"
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
