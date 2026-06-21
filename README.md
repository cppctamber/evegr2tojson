# .about

- `evegr2tojson.exe` - A command line tool for converting EVE Online `.gr2` files to JSON.
- `blender_gr2_loader.py` - A Blender 5.0 plugin for loading `.gr2` and `.gr2_json` files.

## Requirements

`evegr2tojson.exe` is a prebuilt command-line converter for EVE Online `.gr2`
files. It may work with other Granny files from the same Granny version, but
that is untested.

This repository does not include the Granny runtime DLL.

To run the converter, place a compatible `granny2.dll` beside
`evegr2tojson.exe`, or make it available through your local runtime path.

```text
dist/
  evegr2tojson.exe
  granny2.dll          # not included; user-provided local dependency
```

# .usage (windows)

Command line:

```text
evegr2tojson.exe <src> <dest>
```

Blender 5.0 plugin:

1. Download the contents of `/dist`.
2. Load Blender 5.0.
3. Open Edit > Preferences > Addons.
4. From the top right corner `v` icon, select "Install From Disk...".
5. Select the `blender_gr2_loader.py` file.
6. Find the installed plugin and set the `evegr2tojson` file path to where `evegr2tojson.exe` is.
7. File > Import > GR2.

Animations are loaded as Actions: Animation > Action Editor.

# .compile

tba

# .todo

- Mesh min and max bounds
- Convert all animation data into a single normalized format
- Generate normals when missing
- Generate bitangents when missing
- Generate tangents when missing

# .disclaimer

USE THIS AT YOUR OWN RISK. IF YOU USE THIS, YOU AGREE THAT THE AUTHOR CANNOT BE
HELD LIABLE FOR ANY CONSEQUENCES THAT MIGHT OCCUR BECAUSE OF ITS USAGE.

# Licensing Notes

This project is derived from or related to `Helia01/GrannyMeshDumper`, with
additional EVE-specific JSON output behavior.

The bundled executable is provided as-is for local tooling. Third-party runtime
dependencies, including Granny DLLs, must be obtained separately by the user and
used according to their own terms.

# .copyright

EVE Online, the EVE logo, EVE and all associated logos and designs are the
intellectual property of Fenris hf. (formerly CCP hf) All artwork, screenshots, characters, vehicles,
storylines, world facts or other recognizable features of the intellectual
property relating to these trademarks are likewise the intellectual property of
Fenris hf (formerly CCP hf). EVE Online and the EVE logo are the registered trademarks of CCP hf. All
rights are reserved worldwide. All other trademarks are the property of their
respective owners.

CCP is in no way responsible for the content on or functioning of this website,
nor can it be liable for any damage arising from the use of this website.
