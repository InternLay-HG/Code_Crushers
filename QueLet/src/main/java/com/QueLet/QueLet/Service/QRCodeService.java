package com.QueLet.QueLet.Service;

import com.QueLet.QueLet.Model.Appointment;
import com.QueLet.QueLet.Repository.AppointmentRepository;
import com.google.zxing.BarcodeFormat;
import com.google.zxing.WriterException;
import com.google.zxing.client.j2se.MatrixToImageWriter;
import com.google.zxing.common.BitMatrix;
import com.google.zxing.qrcode.QRCodeWriter;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

@Service
public class QRCodeService {
    @Autowired
    private AppointmentRepository appointmentRepository;

    /**
     * Generates a QR code image with a watermark as a byte array.
     *
     * @param width  The width of the QR code image.
     * @param height The height of the QR code image.
     * @return A byte array representing the QR code image with a watermark.
     * @throws WriterException If there is an error generating the QR code.
     * @throws IOException     If there is an error writing the QR code image.
     */
    public byte[] generateQRCodeWithWatermark(int width, int height, Long AppointedId) throws WriterException, IOException {
        // Step 1: Generate the QR code
        Appointment appointment = appointmentRepository.findByid(AppointedId);
        String text = "Appointment ID: " + AppointedId +
                "\nCustomer ID: " + appointment.getCustomer().getUser_id() +
                "\nBusiness ID: " + appointment.getBusiness().getBusinessId() +
                "\nTime: " + appointment.getAppointmentTime();
        String watermarkText = "QueLet";

        QRCodeWriter qrCodeWriter = new QRCodeWriter();
        BitMatrix bitMatrix = qrCodeWriter.encode(text, BarcodeFormat.QR_CODE, width, height);
        BufferedImage qrImage = MatrixToImageWriter.toBufferedImage(bitMatrix);

        // Step 2: Create a new BufferedImage to add the watermark
        BufferedImage combinedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
        Graphics2D graphics = combinedImage.createGraphics();

        // Draw the QR code onto the new BufferedImage
        graphics.drawImage(qrImage, 0, 0, null);

        // Set font and color for watermark
        Font font = new Font("Sans-Serif", Font.BOLD, 25);
        graphics.setFont(font);
        graphics.setColor(new Color(135, 206, 235, 242));  // skyblue color with transparency

        // Calculate the position for the watermark text (centered at the bottom)
        FontMetrics fontMetrics = graphics.getFontMetrics();
        int x = (width - fontMetrics.stringWidth(watermarkText)) / 2;
        int y = (height - fontMetrics.getHeight()) / 2 + fontMetrics.getAscent();

        // Draw the watermark text onto the combined image
        graphics.drawString(watermarkText, x, y);
        graphics.dispose();  // Always dispose of the Graphics2D context

        // Convert the combined image with the watermark to a byte array
        ByteArrayOutputStream pngOutputStream = new ByteArrayOutputStream();
        ImageIO.write(combinedImage, "PNG", pngOutputStream);
        return pngOutputStream.toByteArray();
    }
}