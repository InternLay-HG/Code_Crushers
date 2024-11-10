package com.QueLet.QueLet.Model;

import com.fasterxml.jackson.annotation.JsonBackReference;
import com.fasterxml.jackson.annotation.JsonIgnore;
import jakarta.persistence.*;
import lombok.*;
import org.springframework.security.core.GrantedAuthority;
import org.springframework.security.core.userdetails.UserDetails;

import java.util.Collection;
import java.util.List;

@Entity
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Table(name = "Business")

public class Business implements UserDetails {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long businessId;
    @NonNull
    private String username;
    private String password;
    private String email;
    private String description;
    @Enumerated(EnumType.STRING)
    private CategoryType typeOfBusiness;
    private String businessName;
    private Long gstNumber;
    private List<String> timings;
    private String url;
    @OneToMany(mappedBy = "business")
    //@JsonBackReference
    @JsonIgnore
    private List<Appointment> appointments;
    private int seatsAvailable;

    @Override
    public Collection<? extends GrantedAuthority> getAuthorities() {
        return List.of();
    }

    @Override
    public boolean isAccountNonExpired() {
        return UserDetails.super.isAccountNonExpired();
    }

    @Override
    public boolean isAccountNonLocked() {
        return UserDetails.super.isAccountNonLocked();
    }

    @Override
    public boolean isCredentialsNonExpired() {
        return UserDetails.super.isCredentialsNonExpired();
    }

    @Override
    public boolean isEnabled() {
        return UserDetails.super.isEnabled();
    }
    public Business(Long businessId) {
        this.businessId=businessId;
    }

}
