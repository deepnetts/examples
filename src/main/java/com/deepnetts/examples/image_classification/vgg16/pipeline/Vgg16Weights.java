package com.deepnetts.examples.image_classification.vgg16.pipeline;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

// Gets the VGG16 weights that were trained in TensorFlow: downloads them once (about 600 MB), then reuses the file.
public final class Vgg16Weights {

    private static final URI WEIGHTS_URI = URI.create("https://www.dropbox.com/s/62pr7xyrx2vvpyn/vgg16_imagenet_weights.zip?dl=1");
    private static final String ARCHIVE_FILE = "vgg16_imagenet_weights.zip";
    private static final String WEIGHTS_FILE = "vgg16_imagenet_weights.txt";

    private Vgg16Weights() {
    }

    // Returns the weights file in the cache folder. Downloads and unpacks it only if it is not there yet.
    public static Path obtain(Path cacheFolder) throws IOException {
        Files.createDirectories(cacheFolder);
        Path weights = cacheFolder.resolve(WEIGHTS_FILE);
        if (Files.isRegularFile(weights) && Files.size(weights) > 0) {
            return weights;
        }
        Path archive = cacheFolder.resolve(ARCHIVE_FILE);
        if (!Files.isRegularFile(archive) || Files.size(archive) == 0) {
            System.out.println("  Downloading the VGG16 weights (about 600 MB, only the first time)...");
            download(archive);
        }
        unzip(archive, cacheFolder);
        if (!Files.isRegularFile(weights) || Files.size(weights) == 0) {
            throw new IOException(archive + " does not contain " + WEIGHTS_FILE);
        }
        return weights;
    }

    // Downloads into a .part file first and renames it only when the download is complete,
    // so a broken download is never mistaken for the real file.
    private static void download(Path archive) throws IOException {
        Path partial = archive.resolveSibling(archive.getFileName() + ".part");
        HttpClient client = HttpClient.newBuilder()
                .followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(30))
                .build();
        HttpRequest request = HttpRequest.newBuilder(WEIGHTS_URI).timeout(Duration.ofMinutes(30)).GET().build();
        try {
            HttpResponse<Path> response = client.send(request, HttpResponse.BodyHandlers.ofFile(partial));
            if (response.statusCode() != 200) {
                throw new IOException("Downloading the VGG16 weights failed: HTTP " + response.statusCode());
            }
            try {
                Files.move(partial, archive, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(partial, archive, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Downloading the VGG16 weights was interrupted", e);
        } finally {
            Files.deleteIfExists(partial);
        }
    }

    // Unpacks the archive. A file name inside it may not point outside the cache folder (like ../../file).
    private static void unzip(Path archive, Path folder) throws IOException {
        Path target = folder.toAbsolutePath().normalize();
        try (InputStream input = Files.newInputStream(archive); ZipInputStream zip = new ZipInputStream(input)) {
            ZipEntry entry;
            while ((entry = zip.getNextEntry()) != null) {
                Path output = target.resolve(entry.getName()).normalize();
                if (!output.startsWith(target)) {
                    throw new IOException("Unsafe file name in " + archive.getFileName() + ": " + entry.getName());
                }
                if (entry.isDirectory()) {
                    Files.createDirectories(output);
                } else {
                    Files.createDirectories(output.getParent());
                    Files.copy(zip, output, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }
    }
}
