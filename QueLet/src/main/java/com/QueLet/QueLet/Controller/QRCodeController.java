package com.QueLet.QueLet.Controller;

import com.QueLet.QueLet.Service.QRCodeService;
import com.google.zxing.WriterException;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.IOException;

@RestController
@RequestMapping("/api")
public class QRCodeController {

    @Autowired
    private QRCodeService qrCodeGeneratorService;

    @GetMapping(value = "/generate-Qrcode/{appointmentId}", produces = MediaType.IMAGE_PNG_VALUE)
    public ResponseEntity<byte[]> generateQRCodeWithWatermark( @PathVariable Long appointmentId) {
        try {
            int width = 250;
            int height = 250;
            byte[] qrCodeImage = qrCodeGeneratorService.generateQRCodeWithWatermark( width, height,appointmentId);
            return ResponseEntity.ok(qrCodeImage);
        } catch (WriterException | IOException e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(null);
        }
    }
}