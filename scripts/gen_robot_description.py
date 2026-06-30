#!/usr/bin/env python3
"""Generate a TurtleBot 4 robot_description and strip the world-singleton system plugins.

Why: the stock Create 3 description (irobot_create_description/urdf/create3.urdf.xacro)
declares the GLOBAL, singleton Ignition systems `Sensors` (render) and `Contact` inside
every robot. With one robot that's one declaration (fine); with two it's declared twice and
adding the 2nd robot's sensors to a live scene forces the Sensors system to re-create the
whole scene -> "Visual already exists" / "Parent not found" crash (turtlebot4_simulator#60).

The fix (done in-repo, leaving /opt/ros untouched): declare Sensors + Contact ONCE at world
scope in our sim_world.sdf, and remove them from each robot's description. This script does
the removal as a pure runtime transform: it runs `xacro` on the (unmodified, read-only) stock
description and writes the filtered XML to stdout. It never writes a file; the stock xacro is
only read. PosePublisher and all other plugins are left intact (PosePublisher must stay
per-robot so each namespaced robot publishes its own pose).

Usage (invoked by turtlebot4_spawn_filtered.launch.py via a Command substitution):
    gen_robot_description.py <xacro_file> [xacro_arg ...]
e.g.
    gen_robot_description.py .../turtlebot4.urdf.xacro gazebo:=ignition namespace:=donnie
"""

import re
import subprocess
import sys

# Remove each `<gazebo>` wrapper whose `<plugin>` is one of the world-singleton systems we
# relocate to the world SDF. DOTALL so the multi-line Sensors block (with <render_engine>) is
# captured; non-greedy so we stop at the first </plugin>. Attribute order is irrelevant
# (`[^>]*` spans the whole opening tag, where `filename=` lives).
_STRIP_RE = re.compile(
    r'\s*<gazebo>\s*<plugin\b[^>]*'
    r'libignition-gazebo-(?:sensors|contact)-system\.so'
    r'.*?</plugin>\s*</gazebo>',
    re.DOTALL,
)


def main(argv):
    if len(argv) < 2:
        sys.stderr.write(
            'gen_robot_description.py: expected <xacro_file> [xacro_arg ...]\n')
        return 2

    # Run the stock xacro chain unchanged (read-only). Let xacro errors surface on stderr.
    xml = subprocess.check_output(['xacro', *argv[1:]], text=True)

    sys.stdout.write(_STRIP_RE.sub('', xml))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
