package com.QueLet.QueLet.Service;

import com.QueLet.QueLet.Model.Customer;
import com.QueLet.QueLet.Repository.CustomerRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

@Service
public class CustomerService {
    @Autowired
    private CustomerRepository customerRepository;
    @Autowired
    private PasswordEncoder passwordEncoder;


    public Customer save(Customer user) {
        if ( user.getPassword().isEmpty()) {
            throw new IllegalArgumentException("password cannot be null or empty");
        }
        Optional<Customer> existingUser = customerRepository.findByUsername(user.getUsername());

        if (existingUser.isPresent()) {
            throw new IllegalArgumentException("username already taken");
        }
        user.setPassword(passwordEncoder.encode(user.getPassword()));
        return customerRepository.save(user);
    }

    public List<Customer> AllUser() {
        return customerRepository.findAll();
    }
}
