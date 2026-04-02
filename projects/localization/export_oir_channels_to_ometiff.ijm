inputDir  = getDirectory("Select folder with OIR files");
outputDir = getDirectory("Select output folder for OME-TIFF");

list = getFileList(inputDir);
setBatchMode(true);

for (i = 0; i < list.length; i++) {

    if (endsWith(list[i], ".oir") || endsWith(list[i], ".OIR")) {

        open(inputDir + list[i]);

        baseName = File.getNameWithoutExtension(list[i]);

        // Obtener dimensiones reales (ya correctas)
        Stack.getDimensions(w, h, c, z, t);
        // t == 1 en tu caso

        // Duplicar cada canal completo (todas las Z)
        for (ch = 1; ch <= c; ch++) {

            run("Duplicate...",
                "duplicate channels=" + ch +
                " slices=1-" + z);

            outName = outputDir + baseName + "_CH" + (ch-1) + ".ome.tif";
            saveAs("OME-TIFF", outName);
            close();
        }

        close();
    }
}

setBatchMode(false);
print("Done exporting OIR files.");