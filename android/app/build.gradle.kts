import java.util.Properties
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "me.stedra.stegocam"
    compileSdk = 35

    defaultConfig {
        applicationId = "me.stedra.stegocam"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"
    }

    // Release key lives OUTSIDE the repo: ~/.android/stegocam-release.jks, passwords in ~/.android/stegocam-release.properties (storeFile, storePassword, keyAlias, keyPassword)
    val relProps = Properties().also { p ->
        val f = File(System.getProperty("user.home"), ".android/stegocam-release.properties")
        if (f.exists()) f.inputStream().use { p.load(it) }
    }
    signingConfigs {
        if (relProps.containsKey("storeFile")) create("release") {
            storeFile = file(relProps.getProperty("storeFile")); storePassword = relProps.getProperty("storePassword")
            keyAlias = relProps.getProperty("keyAlias"); keyPassword = relProps.getProperty("keyPassword")
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            // assets: the lab-only mark specs live in src/debug/assets, so the release APK ships the demo cards + sheet refs only
            signingConfig = signingConfigs.findByName("release") ?: signingConfigs.getByName("debug")
        }
    }
    defaultConfig.ndk { abiFilters += "arm64-v8a" }
    ndkVersion = "29.0.14206865"
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.22.1" } }
    defaultConfig.externalNativeBuild { cmake {
        arguments += listOf("-DANDROID_STL=c++_static", "-DCMAKE_BUILD_TYPE=Release")
        cppFlags += "-O3"
    } }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("org.opencv:opencv:4.12.0")
    implementation("com.google.mlkit:barcode-scanning:17.3.0")   // bundled model: no Play-services model download
}
