package com.QueLet.QueLet.Service;

import com.QueLet.QueLet.Model.Appointment;
import com.QueLet.QueLet.Model.Business;
import com.QueLet.QueLet.Model.Customer;
import com.QueLet.QueLet.Repository.AppointmentRepository;
import com.QueLet.QueLet.Repository.BusinessRepository;
import com.QueLet.QueLet.Repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
public class AppointmentService {
    @Autowired
    private AppointmentRepository appointmentRepository;

    @Autowired
    private CustomerRepository customerRepository;

    @Autowired
    private BusinessRepository businessRepository;

    public Appointment addAppointment(Long customerId, Long businessId, LocalDateTime appointmentTime) {
        Customer customer = customerRepository.findById(customerId)
                .orElseThrow(() -> new RuntimeException("Customer not found"));
        Business business = businessRepository.findById(businessId)
                .orElseThrow(() -> new RuntimeException("Business not found"));
        business.setSeatsAvailable(business.getSeatsAvailable() + 1);
        Appointment appointment = new Appointment();
        appointment.setCustomer(customer);
        appointment.setBusiness(business);
        appointment.setAppointmentTime(appointmentTime);


        return appointmentRepository.save(appointment);
    }

    public List<Appointment> findAppointmentsByBusinessId(Long businessId) {
        return appointmentRepository.findByBusinessId(businessId);
    }

    public Appointment completeAppointment(Long appointmentId) {
        Appointment appointment = appointmentRepository.findById(appointmentId)
                .orElseThrow(() -> new RuntimeException("Appointment not found"));
        Business business = appointment.getBusiness();

        // Mark appointment as complete (assuming an `isCompleted` field exists)
        appointment.setIsCompleted(true);
        appointmentRepository.save(appointment);

        // Increment seats available
        business.setSeatsAvailable(business.getSeatsAvailable() - 1);
        businessRepository.save(business);
       return appointmentRepository.save(appointment);
    }
}